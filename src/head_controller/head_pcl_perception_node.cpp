/*
 * head_pcl_perception_node.cpp
 *
 * INDEPENDENT PCL/C++ cross-check pipeline for the head-camera tabletop
 * perception problem, ported from a colleague's tabletop_perception_node
 * (different robot config) into triago_control, per operator instruction to
 * try a structurally different algorithm after our own extrinsic-chain
 * investigation (see triago_control's .kiro/context.md, head_control
 * section) could not find a fixable root cause for a small constant XY bias.
 *
 * WHY THIS IS STRUCTURALLY DIFFERENT FROM main_head.py's PYTHON PIPELINE:
 *   - INPUT: subscribes to the Gazebo RealSense plugin's NATIVE, already
 *     deprojected & coloured point cloud topic (default
 *     "/gripper_head_camera_rgbd/depth/color/points"), instead of our own
 *     manual depth-image deprojection + distortion correction in
 *     camera_interface.py. This bypasses that entire code path, so if the
 *     bias were somewhere in our manual deprojection math, it would NOT
 *     reproduce here.
 *   - SHAPE FIT: uses a GENERIC 2D-PCA oriented-bounding-box (works for any
 *     convex blob), not our cylinder-specific rim-extraction + circle fit.
 *   - TRACKING: nearest-centroid matching with GROW-ONLY dimensions (matches
 *     the colleague's original design exactly) instead of our EMA fusion.
 *
 * BUGS FOUND AND FIXED while porting (both would have failed to build/run):
 *   1. A stray token ("target_pose_publisher_" written on its own line) sat
 *      between two EuclideanClusterExtraction setter calls in the pasted
 *      source -- not valid C++, removed.
 *   2. pcl::EuclideanClusterExtraction was never given a search method
 *      (missing setSearchMethod(KdTree)) -- PCL throws at runtime without
 *      one ("No search method... Please pass one via setSearchMethod()").
 *      Added a KdTree explicitly.
 *
 * ADDITIONS (clearly flagged, not part of the original file):
 *   - A workspace crop (PassThrough on x/y/z) using the SAME known-prior
 *     table location already used everywhere else in this project
 *     (head_control/config.py section 2) -- a legitimate, previously-
 *     established speed/robustness prior, not new scene-specific hacking.
 *   - A VoxelGrid downsample before clustering (the native point cloud can
 *     be dense; RANSAC/clustering on 300k+ raw points per frame would be
 *     slow).
 *   - Colour classification (red/blue/unknown) from the cluster's mean RGB,
 *     needed because head_plotter.py classifies markers by colour channel.
 *   - A read-only bias-vs-range diagnostic against the known simulation
 *     ground truth (mirrors the SAME technique used in main_head.py's
 *     Python pipeline for direct comparison) -- diagnostic console output
 *     ONLY, never fed back into the published perception result, per the
 *     project's standing rule on ground-truth usage.
 *   - A periodic boxed console debug panel (the "debug window" to share).
 *
 * OUTPUT TOPICS (deliberately on a SEPARATE namespace from main_head.py's
 * own /head_perception/markers + /head_perception/telemetry, so both nodes
 * can run at the same time without fighting over the same topic -- run
 * main_head.py for head motion/scanning, and THIS node for perception, then
 * point head_plotter.py at THIS node's topics via a remap, see the
 * accompanying instructions):
 *   /head_perception_pcl/markers      (visualization_msgs/MarkerArray)
 *   /head_perception_pcl/telemetry    (std_msgs/Float64MultiArray, SAME
 *                                       9-float layout head_plotter.py
 *                                       already expects)
 *   /head_perception_pcl/table_cloud  (sensor_msgs/PointCloud2, RViz debug)
 *   /head_perception_pcl/objects_cloud(sensor_msgs/PointCloud2, RViz debug)
 *   /head_perception_pcl/target_pose  (geometry_msgs/PoseStamped)
 *
 * NOT COMPILE-TESTED: this environment has no ROS2/PCL/colcon toolchain
 * available. The code was written and reviewed carefully against the
 * documented ROS 2 Humble + PCL APIs, but a real `colcon build` on the
 * target machine is the first real compile -- see the accompanying message
 * for exactly what to check if it fails.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Dense>

using PointT = pcl::PointXYZRGB;

// ========================================================================
// Known prior / diagnostic constants -- MIRRORED from
// triago_control/head_control/config.py. This project's single source of
// truth for these numbers is that Python file; if the world SDF or crop
// tuning ever changes there, update BOTH copies (this duplication is the
// same pattern already flagged/accepted elsewhere in this codebase, e.g.
// head_control/config.py's own GT_* constants relative to the world_loader
// YAML -- see the project's context.md).
// ========================================================================
namespace known_prior {
// Table location (base_footprint), used ONLY for a workspace crop (speed +
// robustness), exactly like head_control/config.py section 2/7.
constexpr double kTableCenterX = 1.000;
constexpr double kTableCenterY = 0.0;
constexpr double kTableSizeX = 0.6;
constexpr double kTableSizeY = 0.5;
constexpr double kTableTopZ = 0.70;
constexpr double kCropMarginXY = 0.25;
constexpr double kCropZMin = 0.20;
constexpr double kCropZMax = kTableTopZ + 0.45;

// Colour classification thresholds -- mirrors config.py section 11.
constexpr double kColorSatMin = 0.35;
constexpr double kColorValMin = 0.15;
constexpr double kRedHueLow = 0.95;
constexpr double kRedHueHigh = 0.05;
constexpr double kBlueHueLow = 0.55;
constexpr double kBlueHueHigh = 0.75;

// GROUND TRUTH -- READ-ONLY, DIAGNOSTIC USE ONLY (mirrors config.py section
// 13). The perception logic below NEVER reads these; only the console
// bias-vs-range report does, purely for comparison/reporting.
constexpr double kGtRedX = 0.800, kGtRedY = -0.20, kGtRedZ = 0.775;
constexpr double kGtBlueX = 0.800, kGtBlueY = 0.20, kGtBlueZ = 0.775;
}  // namespace known_prior

// ------------------------------------------------------------------------
// Per-object memory (grow-only dimensions, nearest-centroid tracking) --
// same structure/policy as the colleague's original node.
// ------------------------------------------------------------------------
struct TrackedObject {
  int id = -1;
  std::string color_name = "unknown";     // "red" | "blue" | "unknown"
  Eigen::Vector3f position{0, 0, 0};
  Eigen::Quaternionf orientation{1, 0, 0, 0};
  Eigen::Vector3f dimensions{0, 0, 0};     // OBB extents (x, y, z)
  int n_points = 0;
  int frames_unseen = 0;
  bool matched_this_frame = false;
  bool is_obstacle = false;
};

// Read-only diagnostic sample: (range, dx, dy, dz) vs known GT, for the
// bias-vs-range regression report -- see main_head.py's identical technique.
struct BiasSample {
  double range, dx, dy, dz;
};

class HeadPclPerceptionNode : public rclcpp::Node {
 public:
  HeadPclPerceptionNode() : Node("head_pcl_perception_node") {
    target_frame_ = this->declare_parameter<std::string>("base_frame", "base_footprint");
    std::string cloud_topic = this->declare_parameter<std::string>(
        "cloud_topic", "/gripper_head_camera_rgbd/depth/color/points");
    double cluster_tolerance = this->declare_parameter<double>("cluster_tolerance", 0.03);
    int min_cluster_size = this->declare_parameter<int>("min_cluster_size", 50);
    int max_cluster_size = this->declare_parameter<int>("max_cluster_size", 25000);
    double voxel_leaf = this->declare_parameter<double>("voxel_leaf", 0.003);
    double plane_dist_thresh = this->declare_parameter<double>("plane_distance_threshold", 0.02);
    enable_crop_ = this->declare_parameter<bool>("enable_workspace_crop", true);

    cluster_tolerance_ = cluster_tolerance;
    min_cluster_size_ = min_cluster_size;
    max_cluster_size_ = max_cluster_size;
    voxel_leaf_ = voxel_leaf;
    plane_dist_thresh_ = plane_dist_thresh;

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic, rclcpp::SensorDataQoS(),
        std::bind(&HeadPclPerceptionNode::cloudCallback, this, std::placeholders::_1));

    table_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/head_perception_pcl/table_cloud", 1);
    objects_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/head_perception_pcl/objects_cloud", 1);
    // Same topic NAME shape (MarkerArray / Float64MultiArray) and same
    // telemetry LAYOUT as main_head.py -- deliberately on this node's own
    // namespace so both nodes can run concurrently (see file header).
    markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/head_perception_pcl/markers", 1);
    telemetry_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/head_perception_pcl/telemetry", 10);
    target_pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/head_perception_pcl/target_pose", 10);

    console_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(3000),
        std::bind(&HeadPclPerceptionNode::consoleTick, this));

    RCLCPP_INFO(this->get_logger(),
                "\n=================================================================="
                "\n TRIAGo HEAD -- PCL/C++ cross-check perception (colleague's algorithm)"
                "\n------------------------------------------------------------------"
                "\n  Cloud topic : %s"
                "\n  Base frame  : %s"
                "\n  Cluster tol : %.3f m   min/max pts: %d/%d"
                "\n  Voxel leaf  : %.4f m   Workspace crop: %s"
                "\n  Publishing markers/telemetry on /head_perception_pcl/* (a SEPARATE"
                "\n  namespace from main_head.py -- both can run together; remap"
                "\n  head_plotter.py's subscriptions to compare, see run instructions)."
                "\n==================================================================",
                cloud_topic.c_str(), target_frame_.c_str(), cluster_tolerance_,
                min_cluster_size_, max_cluster_size_, voxel_leaf_,
                enable_crop_ ? "ON" : "OFF");
  }

 private:
  static constexpr int kMaxUnseenFrames = 10;              // ~2s at 5Hz-ish input
  static constexpr float kMatchingDistanceThreshold = 0.15f;  // 15cm

  // -------------------------------------------------------------------- //
  // Main callback                                                         //
  // -------------------------------------------------------------------- //
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    const auto t0 = std::chrono::steady_clock::now();

    if (!diag_logged_) {
      std::ostringstream fields;
      for (const auto &f : msg->fields) fields << f.name << " ";
      RCLCPP_INFO(this->get_logger(),
                  "[DIAG] first cloud: frame_id='%s' width=%u height=%u fields=[%s]",
                  msg->header.frame_id.c_str(), msg->width, msg->height,
                  fields.str().c_str());
      diag_logged_ = true;
    }

    // --- 1. TF transform into base_footprint, exact stamp then latest --
    geometry_msgs::msg::TransformStamped transform;
    bool got_tf = false;
    try {
      transform = tf_buffer_->lookupTransform(
          target_frame_, msg->header.frame_id, tf2_ros::fromMsg(msg->header.stamp),
          tf2::durationFromSec(0.05));
      got_tf = true;
    } catch (const tf2::TransformException &) {
      try {
        transform = tf_buffer_->lookupTransform(
            target_frame_, msg->header.frame_id, tf2::TimePointZero,
            tf2::durationFromSec(0.05));
        got_tf = true;
      } catch (const tf2::TransformException &ex2) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                              "Could not transform cloud: %s", ex2.what());
      }
    }
    if (!got_tf) return;

    // Camera origin in base_footprint (for the range-based bias diagnostic).
    const Eigen::Vector3f cam_pos(transform.transform.translation.x,
                                   transform.transform.translation.y,
                                   transform.transform.translation.z);

    sensor_msgs::msg::PointCloud2 transformed_msg;
    try {
      tf2::doTransform(*msg, transformed_msg, transform);
    } catch (const std::exception &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "doTransform failed: %s", ex.what());
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(transformed_msg, *cloud);
    const int n_raw = static_cast<int>(cloud->size());

    // --- 2. (ADDED) workspace crop -- see file header. Uses the SAME
    // known table-location prior already used elsewhere in this project.
    pcl::PointCloud<PointT>::Ptr cropped(new pcl::PointCloud<PointT>());
    if (enable_crop_) {
      pcl::PassThrough<PointT> pass;
      pcl::PointCloud<PointT>::Ptr tmp(new pcl::PointCloud<PointT>());
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(
          known_prior::kTableCenterX - known_prior::kTableSizeX / 2.0 - known_prior::kCropMarginXY,
          known_prior::kTableCenterX + known_prior::kTableSizeX / 2.0 + known_prior::kCropMarginXY);
      pass.filter(*tmp);

      pass.setInputCloud(tmp);
      pass.setFilterFieldName("y");
      pass.setFilterLimits(
          known_prior::kTableCenterY - known_prior::kTableSizeY / 2.0 - known_prior::kCropMarginXY,
          known_prior::kTableCenterY + known_prior::kTableSizeY / 2.0 + known_prior::kCropMarginXY);
      pass.filter(*tmp);

      pass.setInputCloud(tmp);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(known_prior::kCropZMin, known_prior::kCropZMax);
      pass.filter(*cropped);
    } else {
      cropped = cloud;
    }
    const int n_crop = static_cast<int>(cropped->size());

    if (n_crop < min_cluster_size_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Too few points after crop (%d) -- table not in view?", n_crop);
      publishTelemetry(n_raw, n_crop, std::nan(""), 0.0, 0.0, 0.0);
      return;
    }

    // --- 3. (ADDED) voxel downsample before the heavy geometric stages --
    pcl::PointCloud<PointT>::Ptr work(new pcl::PointCloud<PointT>());
    if (voxel_leaf_ > 1e-6) {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cropped);
      vg.setLeafSize(static_cast<float>(voxel_leaf_), static_cast<float>(voxel_leaf_),
                      static_cast<float>(voxel_leaf_));
      vg.filter(*work);
    } else {
      work = cropped;
    }

    // --- 4. RANSAC planar segmentation (colleague's original algorithm) -
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(plane_dist_thresh_);
    seg.setInputCloud(work);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Table not found.");
      publishTelemetry(n_raw, n_crop, std::nan(""), 0.0, 0.0, 0.0);
      return;
    }

    // --- 5. Extract table / objects ------------------------------------
    pcl::PointCloud<PointT>::Ptr cloud_table(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_objects(new pcl::PointCloud<PointT>());

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(work);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_table);
    extract.setNegative(true);
    extract.filter(*cloud_objects);

    PointT min_pt_table, max_pt_table;
    pcl::getMinMax3D(*cloud_table, min_pt_table, max_pt_table);
    const float table_surface_z = max_pt_table.z;

    visualization_msgs::msg::MarkerArray marker_array;
    // table_top: ns/type head_plotter.py actually reads (CUBE, position.z).
    marker_array.markers.push_back(makeTableTopMarker(
        (min_pt_table.x + max_pt_table.x) / 2.0f, (min_pt_table.y + max_pt_table.y) / 2.0f,
        table_surface_z, transformed_msg.header.frame_id));
    // Full 3D bbox, informational only (RViz).
    marker_array.markers.push_back(makeBoundingBoxMarker(
        min_pt_table, max_pt_table, "pcl_table_bbox", 0, 0.0f, 1.0f, 0.0f,
        transformed_msg.header.frame_id, 0.005f));

    // --- 6. Euclidean cluster extraction --------------------------------
    // BUGFIX vs the original pasted source: a KdTree search method is
    // REQUIRED by PCL (missing in the original -> runtime PCL_ERROR); and a
    // stray token that sat between setMaxClusterSize/setInputCloud has been
    // removed (see file header).
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_objects);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_objects);
    ec.extract(cluster_indices);

    for (auto &obj : tracked_objects_) obj.matched_this_frame = false;

    // --- 7. Per-cluster PCA-OBB fit + colour classification + tracking -
    for (const auto &indices : cluster_indices) {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
      for (const auto &idx : indices.indices) cluster->push_back((*cloud_objects)[idx]);
      cluster->width = static_cast<uint32_t>(cluster->size());
      cluster->height = 1;
      cluster->is_dense = true;

      Eigen::Vector4f centroid_3d;
      pcl::compute3DCentroid(*cluster, centroid_3d);

      pcl::PointCloud<PointT>::Ptr cloud_2d(new pcl::PointCloud<PointT>);
      *cloud_2d = *cluster;
      for (auto &p : cloud_2d->points) p.z = 0.0f;

      Eigen::Vector4f centroid_2d;
      pcl::compute3DCentroid(*cloud_2d, centroid_2d);

      Eigen::Matrix3f covariance_3d;
      pcl::computeCovarianceMatrixNormalized(*cloud_2d, centroid_2d, covariance_3d);

      Eigen::Matrix2f covariance_2d = covariance_3d.block<2, 2>(0, 0);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigen_solver(covariance_2d,
                                                                   Eigen::ComputeEigenvectors);
      Eigen::Matrix2f eig_vecs2d = eigen_solver.eigenvectors();

      Eigen::Matrix3f eigenVectorsPCA = Eigen::Matrix3f::Identity();
      eigenVectorsPCA.block<2, 2>(0, 0) = eig_vecs2d;
      eigenVectorsPCA.col(1) =
          eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0)).normalized();

      Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
      projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
      projectionTransform.block<3, 1>(0, 3) =
          -1.f * (projectionTransform.block<3, 3>(0, 0) * centroid_3d.head<3>());

      pcl::PointCloud<PointT>::Ptr cloudPointsProjected(new pcl::PointCloud<PointT>);
      pcl::transformPointCloud(*cluster, *cloudPointsProjected, projectionTransform);

      PointT minPoint, maxPoint;
      pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);

      Eigen::Vector3f dimensions(maxPoint.x - minPoint.x, maxPoint.y - minPoint.y,
                                  maxPoint.z - minPoint.z);

      const Eigen::Vector3f meanDiagonal =
          0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());
      Eigen::Vector3f obb_position = eigenVectorsPCA * meanDiagonal + centroid_3d.head<3>();
      Eigen::Quaternionf obb_orientation(eigenVectorsPCA);

      // (ADDED) colour classification -- needed for head_plotter.py, which
      // classifies markers by colour channel.
      const std::string color_name = classifyColor(*cluster);

      // --- Matching to memory: prefer same-colour match, else nearest --
      float min_dist = 1e9f;
      TrackedObject *best_match = nullptr;
      for (auto &obj : tracked_objects_) {
        if (obj.matched_this_frame) continue;
        const float dist = (obj.position - obb_position).norm();
        if (dist >= kMatchingDistanceThreshold) continue;
        // Prefer a same-colour match over a merely-closer other-colour one
        // (a small robustness addition -- does not alter any numeric
        // position/dimension estimate, only which memory slot a detection
        // is assigned to).
        const bool color_ok =
            (obj.color_name == "unknown" || color_name == "unknown" || obj.color_name == color_name);
        if (color_ok && dist < min_dist) {
          min_dist = dist;
          best_match = &obj;
        }
      }

      if (best_match != nullptr) {
        best_match->position = obb_position;
        best_match->orientation = obb_orientation;
        // GROW-ONLY dimensions -- faithful to the colleague's original
        // design (per operator instruction to try their algorithm as-is).
        best_match->dimensions.x() = std::max(best_match->dimensions.x(), dimensions.x());
        best_match->dimensions.y() = std::max(best_match->dimensions.y(), dimensions.y());
        best_match->dimensions.z() = std::max(best_match->dimensions.z(), dimensions.z());
        best_match->frames_unseen = 0;
        best_match->matched_this_frame = true;
        best_match->n_points = static_cast<int>(cluster->size());
        if (color_name != "unknown") best_match->color_name = color_name;
      } else {
        TrackedObject new_obj;
        new_obj.id = next_marker_id_++;
        new_obj.position = obb_position;
        new_obj.orientation = obb_orientation;
        new_obj.dimensions = dimensions;
        new_obj.frames_unseen = 0;
        new_obj.matched_this_frame = true;
        new_obj.n_points = static_cast<int>(cluster->size());
        new_obj.color_name = color_name;
        tracked_objects_.push_back(new_obj);
      }
    }

    // --- 8. Age / prune, classify obstacle-vs-target, publish markers --
    double red_conf = 0.0, blue_conf = 0.0;
    bool primary_target_published = false;

    for (auto it = tracked_objects_.begin(); it != tracked_objects_.end();) {
      if (!it->matched_this_frame) it->frames_unseen++;

      if (it->frames_unseen >= kMaxUnseenFrames) {
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = transformed_msg.header.frame_id;
        delete_marker.ns = "objects";
        delete_marker.id = it->id;
        delete_marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(delete_marker);
        it = tracked_objects_.erase(it);
        continue;
      }

      it->is_obstacle = (it->position.z() < table_surface_z);

      if (!it->is_obstacle) {
        // Diameter-from-OBB approximation so head_plotter.py's cylinder
        // assumption (scale.x/2 == radius, scale.z == height) stays valid
        // even though the fit itself is a generic rectangular OBB, not a
        // circle -- see file header.
        const float diameter = 0.5f * (it->dimensions.x() + it->dimensions.y());
        const float height = it->dimensions.z();
        const float radius = 0.5f * diameter;

        marker_array.markers.push_back(makeObjectMarker(
            it->id, it->position, radius, height, it->color_name,
            transformed_msg.header.frame_id));

        // Simple heuristic confidence (point-count based) -- NOT the
        // arc-coverage x fit-quality metric used by the Python pipeline
        // (this generic OBB fit has no rim/arc concept). Purely for
        // head_plotter.py telemetry compatibility.
        const double conf = std::clamp(it->n_points / 300.0, 0.0, 1.0);
        if (it->color_name == "red" && it->frames_unseen == 0) red_conf = conf;
        if (it->color_name == "blue" && it->frames_unseen == 0) blue_conf = conf;

        // Read-only bias-vs-GT sample (diagnostic only).
        recordBiasSample(it->color_name, it->position, cam_pos);

        if (!primary_target_published && it->frames_unseen == 0) {
          geometry_msgs::msg::PoseStamped target_pose_msg;
          target_pose_msg.header.frame_id = transformed_msg.header.frame_id;
          target_pose_msg.header.stamp = transformed_msg.header.stamp;
          target_pose_msg.pose.position.x = it->position.x();
          target_pose_msg.pose.position.y = it->position.y();
          target_pose_msg.pose.position.z = it->position.z();
          target_pose_msg.pose.orientation.x = it->orientation.x();
          target_pose_msg.pose.orientation.y = it->orientation.y();
          target_pose_msg.pose.orientation.z = it->orientation.z();
          target_pose_msg.pose.orientation.w = it->orientation.w();
          target_pose_publisher_->publish(target_pose_msg);
          primary_target_published = true;
        }
      } else {
        const float diameter = 0.5f * (it->dimensions.x() + it->dimensions.y());
        marker_array.markers.push_back(makeObjectMarker(
            it->id, it->position, 0.5f * diameter, it->dimensions.z(), "obstacle",
            transformed_msg.header.frame_id));
      }
      ++it;
    }

    // --- 9. Publish clouds + markers + telemetry ------------------------
    sensor_msgs::msg::PointCloud2 table_msg, objects_msg;
    pcl::toROSMsg(*cloud_table, table_msg);
    pcl::toROSMsg(*cloud_objects, objects_msg);
    table_msg.header = transformed_msg.header;
    objects_msg.header = transformed_msg.header;
    table_publisher_->publish(table_msg);
    objects_publisher_->publish(objects_msg);
    markers_publisher_->publish(marker_array);

    const auto t1 = std::chrono::steady_clock::now();
    const double proc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    publishTelemetry(n_raw, n_crop, table_surface_z, proc_ms, red_conf, blue_conf);

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      last_n_raw_ = n_raw;
      last_n_crop_ = n_crop;
      last_plane_z_ = table_surface_z;
      last_proc_ms_ = proc_ms;
      last_frame_id_ = transformed_msg.header.frame_id;
      last_cam_pos_ = cam_pos;
      have_data_ = true;
    }
  }

  // -------------------------------------------------------------------- //
  // Telemetry -- SAME 9-float layout head_plotter.py expects:            //
  // [n_raw, n_crop, plane_z, look_err_deg, slack, proc_ms,                //
  //  red_conf, blue_conf, map_size]                                       //
  // This node has no head-motion controller, so look_err_deg/slack/      //
  // map_size are always 0 (dummy) -- head_plotter.py tolerates this fine, //
  // those fields simply plot as flat lines.                               //
  // -------------------------------------------------------------------- //
  void publishTelemetry(int n_raw, int n_crop, double plane_z, double proc_ms, double red_conf,
                         double blue_conf) {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {static_cast<double>(n_raw), static_cast<double>(n_crop), plane_z,
                0.0, 0.0, proc_ms, red_conf, blue_conf, 0.0};
    telemetry_publisher_->publish(msg);
  }

  // -------------------------------------------------------------------- //
  // Colour classification (HSV, mirrors head_control/config.py section 11)//
  // -------------------------------------------------------------------- //
  static std::string classifyColor(const pcl::PointCloud<PointT> &cluster) {
    if (cluster.empty()) return "unknown";
    double r = 0, g = 0, b = 0;
    for (const auto &p : cluster.points) {
      r += p.r;
      g += p.g;
      b += p.b;
    }
    const double n = static_cast<double>(cluster.size());
    r /= n; g /= n; b /= n;   // 0..255

    const double rn = r / 255.0, gn = g / 255.0, bn = b / 255.0;
    const double maxc = std::max({rn, gn, bn});
    const double minc = std::min({rn, gn, bn});
    const double v = maxc;
    const double s = (maxc < 1e-9) ? 0.0 : (maxc - minc) / maxc;
    double h = 0.0;
    const double delta = maxc - minc;
    if (delta > 1e-9) {
      if (maxc == rn) {
        h = std::fmod((gn - bn) / delta, 6.0);
      } else if (maxc == gn) {
        h = (bn - rn) / delta + 2.0;
      } else {
        h = (rn - gn) / delta + 4.0;
      }
      h /= 6.0;
      if (h < 0.0) h += 1.0;
    }

    if (s < known_prior::kColorSatMin || v < known_prior::kColorValMin) return "unknown";
    if (h >= known_prior::kRedHueLow || h <= known_prior::kRedHueHigh) return "red";
    if (h >= known_prior::kBlueHueLow && h <= known_prior::kBlueHueHigh) return "blue";
    return "unknown";
  }

  // -------------------------------------------------------------------- //
  // Read-only bias-vs-GT accumulation + regression (mirrors main_head.py) //
  // -------------------------------------------------------------------- //
  void recordBiasSample(const std::string &color_name, const Eigen::Vector3f &pos,
                         const Eigen::Vector3f &cam_pos) {
    double gt_x, gt_y, gt_z;
    if (color_name == "red") {
      gt_x = known_prior::kGtRedX; gt_y = known_prior::kGtRedY; gt_z = known_prior::kGtRedZ;
    } else if (color_name == "blue") {
      gt_x = known_prior::kGtBlueX; gt_y = known_prior::kGtBlueY; gt_z = known_prior::kGtBlueZ;
    } else {
      return;
    }
    const double range = (pos - cam_pos).norm();
    const double dx = pos.x() - gt_x;
    const double dy = pos.y() - gt_y;
    const double dz = pos.z() - gt_z;

    std::lock_guard<std::mutex> lock(state_mutex_);
    auto &buf = bias_samples_[color_name];
    buf.push_back({range, dx, dy, dz});
    if (buf.size() > 500) buf.pop_front();
  }

  // Simple closed-form least-squares line fit: y = intercept + slope * x.
  static std::pair<double, double> linreg(const std::vector<double> &x,
                                           const std::vector<double> &y) {
    const size_t n = x.size();
    if (n < 2) return {y.empty() ? 0.0 : y[0], 0.0};
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) {
      sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    const double denom = n * sxx - sx * sx;
    if (std::abs(denom) < 1e-12) return {sy / n, 0.0};
    const double slope = (n * sxy - sx * sy) / denom;
    const double intercept = (sy - slope * sx) / n;
    return {intercept, slope};
  }

  // -------------------------------------------------------------------- //
  // Marker builders                                                       //
  // -------------------------------------------------------------------- //
  static visualization_msgs::msg::Marker makeTableTopMarker(float cx, float cy, float z,
                                                              const std::string &frame_id) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.ns = "table_top";  // head_plotter.py reads THIS ns + CUBE type + position.z
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = cx;
    m.pose.position.y = cy;
    m.pose.position.z = z;
    m.pose.orientation.w = 1.0;
    m.scale.x = known_prior::kTableSizeX;
    m.scale.y = known_prior::kTableSizeY;
    m.scale.z = 0.005;
    m.color.r = 0.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 0.4f;
    return m;
  }

  static visualization_msgs::msg::Marker makeBoundingBoxMarker(
      const PointT &min_pt, const PointT &max_pt, const std::string &ns, int id, float r,
      float g, float b, const std::string &frame_id, float padding) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = (max_pt.x + min_pt.x) / 2.0;
    marker.pose.position.y = (max_pt.y + min_pt.y) / 2.0;
    marker.pose.position.z = (max_pt.z + min_pt.z) / 2.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = max_pt.x - min_pt.x + padding;
    marker.scale.y = max_pt.y - min_pt.y + padding;
    marker.scale.z = max_pt.z - min_pt.z + padding;
    marker.color.r = r; marker.color.g = g; marker.color.b = b; marker.color.a = 0.5f;
    return marker;
  }

  // ns="objects", type=CYLINDER -- exactly what head_plotter.py looks for
  // (m.ns=="objects" and m.type==3), classifying red/blue by colour channel.
  static visualization_msgs::msg::Marker makeObjectMarker(int id, const Eigen::Vector3f &pos,
                                                            float radius, float height,
                                                            const std::string &color_name,
                                                            const std::string &frame_id) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.ns = "objects";
    m.id = id;
    m.type = visualization_msgs::msg::Marker::CYLINDER;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = pos.x();
    m.pose.position.y = pos.y();
    m.pose.position.z = pos.z();
    m.pose.orientation.w = 1.0;   // identity -- see file header on why not the PCA orientation
    m.scale.x = 2.0f * radius;
    m.scale.y = 2.0f * radius;
    m.scale.z = std::max(height, 0.01f);
    if (color_name == "red") {
      m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f;
    } else if (color_name == "blue") {
      m.color.r = 0.0f; m.color.g = 0.0f; m.color.b = 1.0f;
    } else {
      m.color.r = 1.0f; m.color.g = 0.5f; m.color.b = 0.0f;  // obstacle/unknown -> orange
    }
    m.color.a = 0.6f;
    return m;
  }

  // -------------------------------------------------------------------- //
  // Console debug panel (the "debug window" to screenshot/share)          //
  // -------------------------------------------------------------------- //
  void consoleTick() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!have_data_) {
      RCLCPP_INFO(this->get_logger(), "Waiting for first point cloud frame...");
      return;
    }

    std::ostringstream out;
    out << "\n       +== HEAD PCL PERCEPTION (colleague's algorithm, C++/PCL) ======+\n";
    out << "       | raw=" << last_n_raw_ << " crop=" << last_n_crop_
        << " | plane_z=" << std::fixed << std::setprecision(3) << last_plane_z_
        << " m | proc=" << std::setprecision(1) << last_proc_ms_ << " ms"
        << std::string(std::max(0, 12), ' ') << "|\n";
    out << "       |----------------------------------------------------------------|\n";
    out << "       | tracked objects:                                                |\n";
    if (tracked_objects_.empty()) {
      out << "       |   (none)                                                        |\n";
    }
    for (const auto &o : tracked_objects_) {
      const float diameter = 0.5f * (o.dimensions.x() + o.dimensions.y());
      out << "       |   id=" << o.id << " " << std::left << std::setw(8) << o.color_name
          << " pos=(" << std::fixed << std::setprecision(2) << o.position.x() << ","
          << o.position.y() << "," << o.position.z() << ") r=" << std::setprecision(1)
          << (diameter * 50.0f) << "cm h=" << (o.dimensions.z() * 100.0f)
          << "cm n=" << o.n_points << " unseen=" << o.frames_unseen << " |\n";
    }
    out << "       |-- BIAS-VS-GT REGRESSION (raw detections, read-only) ------------|\n";
    for (const std::string color : {"red", "blue"}) {
      auto it = bias_samples_.find(color);
      if (it == bias_samples_.end() || it->second.size() < 5) {
        out << "       |   " << color << ": not enough samples yet"
            << std::string(30, ' ') << "|\n";
        continue;
      }
      const auto &buf = it->second;
      // z-bias is not rendered in this panel (height is already validated
      // as accurate by the Python pipeline) -- only x/y are collected here.
      std::vector<double> rng, dx, dy;
      for (const auto &s : buf) {
        rng.push_back(s.range); dx.push_back(s.dx); dy.push_back(s.dy);
      }
      auto [ix, sx] = linreg(rng, dx);
      auto [iy, sy] = linreg(rng, dy);
      auto verdict = [](double slope) { return std::abs(slope) < 0.01 ? "CONST" : "SCALES"; };
      out << "       |   " << std::left << std::setw(5) << color
          << " dx: " << std::showpos << std::setprecision(2) << ix * 100 << "cm  slope "
          << sx * 100 << "cm/m [" << verdict(sx) << "] n=" << buf.size() << std::noshowpos
          << std::string(6, ' ') << "|\n";
      out << "       |   " << std::string(5, ' ')
          << " dy: " << std::showpos << iy * 100 << "cm  slope " << sy * 100
          << "cm/m [" << verdict(sy) << "]" << std::noshowpos << std::string(18, ' ') << "|\n";
    }
    out << "       +================================================================+";
    RCLCPP_INFO(this->get_logger(), "%s", out.str().c_str());
  }

  // -------------------------------------------------------------------- //
  // Members                                                               //
  // -------------------------------------------------------------------- //
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr table_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr objects_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr telemetry_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_publisher_;
  rclcpp::TimerBase::SharedPtr console_timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string target_frame_;

  double cluster_tolerance_ = 0.03;
  int min_cluster_size_ = 50;
  int max_cluster_size_ = 25000;
  double voxel_leaf_ = 0.003;
  double plane_dist_thresh_ = 0.02;
  bool enable_crop_ = true;
  bool diag_logged_ = false;

  std::vector<TrackedObject> tracked_objects_;
  int next_marker_id_ = 10;

  std::mutex state_mutex_;
  bool have_data_ = false;
  int last_n_raw_ = 0, last_n_crop_ = 0;
  double last_plane_z_ = 0.0, last_proc_ms_ = 0.0;
  std::string last_frame_id_;
  Eigen::Vector3f last_cam_pos_{0, 0, 0};
  std::map<std::string, std::deque<BiasSample>> bias_samples_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HeadPclPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
