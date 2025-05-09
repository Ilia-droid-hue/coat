# coating_simulator_project/coating_simulator/visualization/geom_emission_linear.py
"""
Functions for plotting emission details specifically for a LINEAR source.
Emission footprint calculation now correctly considers source orientation (tilt).
Side view emission envelope calculation also corrected to account for tilt.
Handles 90-degree emission angle more robustly.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback

from ..core.distribution import rotation_matrix 
from .. import config 
# Import helpers from geom_utils
from .geom_utils import _convex_hull, _is_point_in_polygon, _generate_substrate_boundary
# Import helpers from geom_emission
from .geom_emission import _intersect_z0, get_rotation_matrix_align_vectors

def plot_linear_emission(ax_top: plt.Axes, ax_side: plt.Axes,
                         params: dict, common_data: dict) -> None:
    """
    Plots emission for a linear source, considering its length, orientation (rot_x, rot_y),
    and local rotation (src_angle) for a more accurate emission footprint and side view envelope.
    Handles 90-degree emission angle more robustly.
    """
    try:
        # Source definition from params
        source_actual_length = params.get('src_length', 100.0)
        src_angle_deg = params.get('src_angle', 0.0) 

        # Source position and main orientation from common_data
        source_center_global = common_data.get('source_center_global')
        source_orient_matrix = common_data.get('source_orient_matrix')
        emission_axis_global = common_data.get('emission_axis_global') # Main emission direction

        if source_center_global is None or source_orient_matrix is None or emission_axis_global is None:
             print("ERROR (linear_emission): Missing common_data (center, orientation, or axis).")
             return
        # Ensure vectors are 1D
        if source_center_global.shape != (3,): source_center_global = source_center_global.flatten()
        if emission_axis_global.shape != (3,): emission_axis_global = emission_axis_global.flatten()
        if source_center_global.shape != (3,) or emission_axis_global.shape != (3,):
             print("ERROR (linear_emission): Invalid shape for common_data vectors.")
             return

        # Emission characteristic
        max_theta_rad = common_data.get('max_theta_rad', math.radians(30.0))
        full_angle_deg_for_label = common_data.get('full_angle_deg_for_label', math.degrees(max_theta_rad*2))
        
        # Substrate definition
        target_type = params.get('target_type', config.TARGET_DISK) 
        
        # Styling from common_data
        emission_color = common_data['colors'].get('emission', 'orange')
        emission_fill_color = common_data['colors'].get('emission_fill', emission_color)
        emission_area_alpha = common_data['alphas'].get('emission_area_fill', 0.4) 
        polygon_outline_linewidth = common_data['linewidths'].get('polygon_outline', 1.2)
        emission_linestyle = '--' 
        source_line_color = common_data['colors'].get('source_main', 'darkred') 
        source_line_linewidth = common_data['linewidths'].get('source_main', 2.0)

        # --- Calculations for 3D source geometry ---
        half_src_len = source_actual_length / 2.0
        A_local_initial = np.array([-half_src_len, 0.0, 0.0]) 
        B_local_initial = np.array([ half_src_len, 0.0, 0.0]) 
        src_angle_rad = math.radians(src_angle_deg)
        R_src_angle_local_z = rotation_matrix(np.array([0,0,1]), src_angle_rad)
        A_local_oriented = R_src_angle_local_z.dot(A_local_initial)
        B_local_oriented = R_src_angle_local_z.dot(B_local_initial)
        A_global = source_center_global + source_orient_matrix.dot(A_local_oriented)
        B_global = source_center_global + source_orient_matrix.dot(B_local_oriented)

        # --- Top View (XY Plane) ---
        # (Top view logic remains the same as in geom_emission_linear_py_fix_tilt)
        A_xy = A_global[:2] 
        B_xy = B_global[:2] 
        z_A = A_global[2]
        z_B = B_global[2]
        
        is_angle_90_deg = abs(max_theta_rad - math.pi / 2.0) < 1e-7 
        draw_top_emission_polygon = True
        
        avg_z = (z_A + z_B) / 2.0
        if is_angle_90_deg and avg_z > 1e-3: 
            draw_top_emission_polygon = False 
            print("INFO (linear_emission): Emission angle is ~90 deg and source is above Z=0. Top emission footprint polygon not drawn.")

        all_spread_intersection_xy_points = []
        if draw_top_emission_polygon:
            num_source_samples = 11 
            source_sample_points_global = [A_global + (B_global - A_global) * t 
                                           for t in np.linspace(0, 1, num_source_samples)]
            num_cone_boundary_rays = 16 
            plot_xlim_top, plot_ylim_top = ax_top.get_xlim(), ax_top.get_ylim()
            span_x_top = abs(plot_xlim_top[1] - plot_xlim_top[0]) if abs(plot_xlim_top[1] - plot_xlim_top[0]) > 1e-3 else 1000.0
            span_y_top = abs(plot_ylim_top[1] - plot_ylim_top[0]) if abs(plot_ylim_top[1] - plot_ylim_top[0]) > 1e-3 else 1000.0
            far_dist_top = max(span_x_top, span_y_top) * 2.0
            R_align_cone_axis = get_rotation_matrix_align_vectors(np.array([0,0,-1.0]), emission_axis_global)

            for current_source_point_global in source_sample_points_global: 
                phi_cone_rays = np.linspace(0, 2*np.pi, num_cone_boundary_rays, endpoint=False)
                for phi_cr in phi_cone_rays: 
                    dir_local_on_cone_edge = np.array([math.sin(max_theta_rad) * math.cos(phi_cr),
                                                       math.sin(max_theta_rad) * math.sin(phi_cr),
                                                       -math.cos(max_theta_rad)]) 
                    spread_ray_global_dir = R_align_cone_axis.dot(dir_local_on_cone_edge) 
                    
                    if spread_ray_global_dir.shape != (3,):
                         print(f"ERROR (linear_emission top spread dir): spread_ray_global_dir shape is {spread_ray_global_dir.shape}. Skipping ray.")
                         continue

                    if is_angle_90_deg: 
                        if abs(current_source_point_global[2]) < 1e-3 : 
                           xy_dir = spread_ray_global_dir[:2]
                           norm_xy_dir = np.linalg.norm(xy_dir)
                           if norm_xy_dir > 1e-7:
                               far_point = current_source_point_global[:2] + (xy_dir / norm_xy_dir) * far_dist_top
                               all_spread_intersection_xy_points.append(far_point) 
                    else: 
                        intersect_pt = _intersect_z0(current_source_point_global, spread_ray_global_dir)
                        if intersect_pt is not None: 
                            all_spread_intersection_xy_points.append(intersect_pt[:2]) 

            emission_footprint_on_z0_vertices = []
            if all_spread_intersection_xy_points:
                intersection_points_tuples = [tuple(p) for p in all_spread_intersection_xy_points]
                if len(intersection_points_tuples) >= 3:
                    try:
                        emission_footprint_on_z0_vertices = _convex_hull(intersection_points_tuples)
                    except Exception as e_hull:
                        print(f"ERROR (linear_emission ConvexHull): {e_hull}. Plotting raw points.")
                        ax_top.plot(np.array(intersection_points_tuples)[:,0], 
                                    np.array(intersection_points_tuples)[:,1], 
                                    '.', color=emission_color, alpha=0.3, markersize=2, label='Точки эмиссии (ошибка hull)')
                        emission_footprint_on_z0_vertices = [] 

            substrate_vertices_xy = _generate_substrate_boundary(target_type, params) 
            
            final_intersection_polygon_vertices = [] 
            if emission_footprint_on_z0_vertices and substrate_vertices_xy:
                for pt_emission_tuple in emission_footprint_on_z0_vertices:
                    if _is_point_in_polygon(pt_emission_tuple, substrate_vertices_xy):
                        final_intersection_polygon_vertices.append(pt_emission_tuple)
                for pt_substrate_tuple in substrate_vertices_xy:
                    if _is_point_in_polygon(pt_substrate_tuple, emission_footprint_on_z0_vertices):
                        final_intersection_polygon_vertices.append(pt_substrate_tuple)
                
                if final_intersection_polygon_vertices and len(final_intersection_polygon_vertices) >=3:
                    final_intersection_polygon_vertices = _convex_hull(final_intersection_polygon_vertices) 
                    if final_intersection_polygon_vertices and len(final_intersection_polygon_vertices) >=3:
                        intersection_patch_top = patches.Polygon(
                            np.array(final_intersection_polygon_vertices), 
                            facecolor=emission_fill_color, edgecolor=emission_color,
                            alpha=emission_area_alpha, linewidth=polygon_outline_linewidth,
                            linestyle=emission_linestyle, zorder=5, label='Область эмиссии (Z=0)')
                        ax_top.add_patch(intersection_patch_top)
                elif not final_intersection_polygon_vertices and emission_footprint_on_z0_vertices:
                     pass 

        ax_top.plot([A_xy[0], B_xy[0]], [A_xy[1], B_xy[1]], 
                    color=source_line_color, linewidth=source_line_linewidth * 0.75, 
                    linestyle='-', zorder=6, label='Источник (проекция XY)')

        # --- Side View (XZ Plane) ---
        xA, zA_src = A_global[0], A_global[2] 
        xB, zB_src = B_global[0], B_global[2] 

        side_emission_vertices_xz_tuples = []

        # Calculate the angle of the main emission axis in the XZ plane
        emission_axis_xz = np.array([emission_axis_global[0], emission_axis_global[2]])
        if np.linalg.norm(emission_axis_xz) < 1e-9:
             # If axis is vertical (or nearly), angle is ambiguous, treat as pointing down
             angle_main_axis_xz = -math.pi / 2.0 
        else:
             # Angle from positive X axis, counter-clockwise
             # atan2(y, x) -> atan2(z, x) for XZ plane angle from +X axis
             # We want angle from +Z axis, clockwise for standard math angle
             # Or angle from -Z axis, counter-clockwise like before
             angle_main_axis_xz = math.atan2(emission_axis_global[0], -emission_axis_global[2]) if abs(emission_axis_global[2]) > 1e-7 else \
                                  (math.pi/2 if emission_axis_global[0] > 0 else -math.pi/2)

        if is_angle_90_deg:
            # For 90 deg, draw lines parallel to XY plane from source points
            xlim_side = ax_side.get_xlim() 
            if abs(xlim_side[1] - xlim_side[0]) < 1e-3: 
                current_x_center = (xA + xB) / 2
                xlim_side = (current_x_center - 100, current_x_center + 100) 

            temp_side_pts = [
                (xA, zA_src), (xB, zB_src),
                (xlim_side[1], zB_src), (xlim_side[1], zA_src), 
                (xlim_side[0], zA_src), (xlim_side[0], zB_src)
            ]
            side_emission_vertices_xz_tuples = _convex_hull([ (float(pt[0]), float(pt[1])) for pt in temp_side_pts])
        else: 
            # Calculate the two boundary directions in the XZ plane
            angle1_xz = angle_main_axis_xz - max_theta_rad
            angle2_xz = angle_main_axis_xz + max_theta_rad
            
            # Directions are (sin(angle), -cos(angle)) for angle relative to -Z axis
            dir1_xz = np.array([math.sin(angle1_xz), -math.cos(angle1_xz)])
            dir2_xz = np.array([math.sin(angle2_xz), -math.cos(angle2_xz)])

            # Find intersections for rays starting from A and B
            intersect_A1 = _intersect_z0(A_global, np.array([dir1_xz[0], 0, dir1_xz[1]])) # Add dummy Y=0
            intersect_A2 = _intersect_z0(A_global, np.array([dir2_xz[0], 0, dir2_xz[1]]))
            intersect_B1 = _intersect_z0(B_global, np.array([dir1_xz[0], 0, dir1_xz[1]]))
            intersect_B2 = _intersect_z0(B_global, np.array([dir2_xz[0], 0, dir2_xz[1]]))

            # Collect valid intersection points (XZ coordinates)
            intersections_z0_xz = []
            if intersect_A1 is not None: intersections_z0_xz.append(intersect_A1[[0, 2]])
            if intersect_A2 is not None: intersections_z0_xz.append(intersect_A2[[0, 2]])
            if intersect_B1 is not None: intersections_z0_xz.append(intersect_B1[[0, 2]])
            if intersect_B2 is not None: intersections_z0_xz.append(intersect_B2[[0, 2]])

            if len(intersections_z0_xz) >= 2:
                 # Find the min and max X among intersection points
                 min_x_z0 = min(p[0] for p in intersections_z0_xz)
                 max_x_z0 = max(p[0] for p in intersections_z0_xz)
                 # Vertices: source points + outermost intersections at Z=0
                 temp_side_pts_tuples = [
                     (float(xA), float(zA_src)),
                     (float(xB), float(zB_src)),
                     (float(max_x_z0), 0.0), 
                     (float(min_x_z0), 0.0)  
                 ]
                 side_emission_vertices_xz_tuples = _convex_hull(temp_side_pts_tuples)
            elif len(intersections_z0_xz) == 1: # Only one intersection, forms triangle
                 temp_side_pts_tuples = [
                     (float(xA), float(zA_src)),
                     (float(xB), float(zB_src)),
                     (float(intersections_z0_xz[0][0]), 0.0)
                 ]
                 side_emission_vertices_xz_tuples = _convex_hull(temp_side_pts_tuples)
            else: # No intersections (e.g., source below Z=0 and emitting upwards)
                 side_emission_vertices_xz_tuples = []


        # Draw the side view polygon if vertices were found
        if side_emission_vertices_xz_tuples and len(side_emission_vertices_xz_tuples) >= 3:
            try:
                emission_polygon_side = patches.Polygon(
                    np.array(side_emission_vertices_xz_tuples), 
                    facecolor=emission_fill_color, edgecolor=emission_color,
                    alpha=emission_area_alpha, linewidth=polygon_outline_linewidth,
                    linestyle=emission_linestyle, zorder=5)
                ax_side.add_patch(emission_polygon_side)
            except Exception as e_side_poly:
                print(f"ERROR (linear_emission side polygon): {e_side_poly}")
                traceback.print_exc()

        # Draw the source line itself in the side view
        ax_side.plot([xA, xB], [zA_src, zB_src],
                     color=source_line_color, linewidth=source_line_linewidth,
                     linestyle='-', zorder=6, label='Источник (вид сбоку XZ)')
        
        # Add label for emission angle to side view (optional)
        # Find a representative point to place the label near
        mid_x_src = (xA + xB) / 2
        mid_z_src = (zA_src + zB_src) / 2
        # ax_side.text(mid_x_src, mid_z_src + 5, f"{full_angle_deg_for_label:.0f}°", 
        #              ha='center', va='bottom', color=emission_color, fontsize='small')


    except Exception as e:
        print(f"ERROR in plot_linear_emission: {e}")
        traceback.print_exc()

# Keep the __main__ block for standalone testing if desired
if __name__ == '__main__':
    # ... (rest of the __main__ block from previous version) ...
     pass # Keep the test code if you use it
