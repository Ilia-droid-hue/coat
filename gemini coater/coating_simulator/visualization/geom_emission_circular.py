# coating_simulator_project/coating_simulator/visualization/geom_emission_circular.py
"""
Functions for plotting emission details specifically for a CIRCULAR source.
Correctly calculates and draws the intersection of the emission footprint with the substrate.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback 

# Используем относительные импорты
from .geom_emission import _intersect_z0, get_rotation_matrix_align_vectors
# --- ИМПОРТ ИЗ geom_utils.py ---
from .geom_utils import _convex_hull, _is_point_in_polygon, _generate_substrate_boundary
# ------------------------------
from .geom_source import _apply_rotation 
from .. import config 

def plot_circular_emission(ax_top: plt.Axes, ax_side: plt.Axes, params: dict, common_data: dict):
    """
    Plots emission for a circular source, drawing the intersection with the substrate.
    Side view: Central dashed axis and two spread lines from the source edges ("truncated cone").
    Top view: Filled smooth outline of the emission intersection area with dashed border.
    Отображает эмиссию для круглого источника, рисуя пересечение с подложкой.
    Вид сбоку: Центральная пунктирная ось и две линии разлета от краев источника ("усеченный конус").
    Вид сверху: Залитый сглаженный контур области пересечения эмиссии с пунктирной границей.
    """
    try:
        source_center_global = common_data.get('source_center_global')
        source_orient_matrix = common_data.get('source_orient_matrix')
        emission_axis_global = common_data.get('emission_axis_global') 
        max_theta_rad = common_data.get('max_theta_rad', math.radians(30.0))
        full_angle_deg_for_label = common_data.get('full_angle_deg_for_label', math.degrees(max_theta_rad*2))
        
        # Ensure vectors are 1D
        if source_center_global.shape != (3,): source_center_global = source_center_global.flatten()
        if emission_axis_global.shape != (3,): emission_axis_global = emission_axis_global.flatten()
        if source_center_global.shape != (3,) or emission_axis_global.shape != (3,):
             print("ERROR (circular_emission): Invalid shape for common_data vectors.")
             return
             
        # Styling
        emission_color = common_data['colors'].get('emission', 'orange') 
        emission_fill_color = common_data['colors'].get('emission_fill', emission_color) 
        emission_area_alpha = common_data['alphas'].get('emission_area_fill', 0.4) 
        general_line_alpha = common_data['alphas'].get('general_line', 0.75) 
        circular_edge_spread_alpha = common_data['alphas'].get('circular_edge_spread', 0.65) 
        polygon_outline_linewidth = common_data['linewidths'].get('polygon_outline', 1.2)
        circular_edge_spread_linewidth = common_data['linewidths'].get('circular_edge_spread', 1.0)
        emission_linestyle = '--' 

        is_angle_90_deg = abs(max_theta_rad - math.pi / 2.0) < 1e-7

        # --- Вид сбоку (Side View) ---
        # (Side view logic remains the same - it shows spread from edges)
        angle_of_main_axis_xz = math.atan2(emission_axis_global[0], -emission_axis_global[2]) \
                                  if abs(emission_axis_global[2]) > 1e-7 else \
                                  (math.pi/2 if emission_axis_global[0] > 0 else -math.pi/2)
        dir_spread1_global_cone_edge_xz = np.array([math.sin(angle_of_main_axis_xz - max_theta_rad), 0, 
                                                 -math.cos(angle_of_main_axis_xz - max_theta_rad)])
        dir_spread2_global_cone_edge_xz = np.array([math.sin(angle_of_main_axis_xz + max_theta_rad), 0, 
                                                 -math.cos(angle_of_main_axis_xz + max_theta_rad)])
        
        src_diameter_circ = params.get('src_diameter', 10.0)
        src_radius_circ = src_diameter_circ / 2.0
        
        xlim_side_plot = ax_side.get_xlim()
        plot_span_x_side = abs(xlim_side_plot[1] - xlim_side_plot[0])
        far_x_neg_side = xlim_side_plot[0] - 0.1 * plot_span_x_side 
        far_x_pos_side = xlim_side_plot[1] + 0.1 * plot_span_x_side

        if src_radius_circ > 1e-6: # Draw spread from edges if source has size
            _local_edge_points_circ_side_vis = [np.array([src_radius_circ, 0, 0]), np.array([-src_radius_circ, 0, 0])]
            _p1_y_edge_g = _apply_rotation(np.array([0, src_radius_circ, 0]), source_orient_matrix, source_center_global)
            _p2_y_edge_g = _apply_rotation(np.array([0, -src_radius_circ, 0]), source_orient_matrix, source_center_global)
            _p1_x_edge_g = _apply_rotation(np.array([src_radius_circ, 0, 0]), source_orient_matrix, source_center_global)
            _p2_x_edge_g = _apply_rotation(np.array([-src_radius_circ, 0, 0]), source_orient_matrix, source_center_global)
            
            local_edges_for_side_view = _local_edge_points_circ_side_vis # Default
            if abs(_p1_y_edge_g[0] - _p2_y_edge_g[0]) > abs(_p1_x_edge_g[0] - _p2_x_edge_g[0]):
                 local_edges_for_side_view = [np.array([0, src_radius_circ, 0]), np.array([0, -src_radius_circ, 0])]
            
            global_edge_points_circ_side_vis = [_apply_rotation(p_loc, source_orient_matrix, source_center_global) 
                                                for p_loc in local_edges_for_side_view]

            # Sort edges by X coordinate for consistent plotting
            sorted_edges_circ_side_vis = sorted(global_edge_points_circ_side_vis, key=lambda p: p[0])
            start_edge1_circ_vis = sorted_edges_circ_side_vis[0] # Leftmost edge point in XZ view
            start_edge2_circ_vis = sorted_edges_circ_side_vis[1] # Rightmost edge point in XZ view

            # Calculate endpoints for spread lines from edges
            end_points_side = [None, None]
            spread_dirs_side = [dir_spread1_global_cone_edge_xz, dir_spread2_global_cone_edge_xz]
            start_points_side = [start_edge1_circ_vis, start_edge2_circ_vis]

            for i_side, dir_s_side in enumerate(spread_dirs_side):
                start_p_side = start_points_side[i_side]
                if abs(dir_s_side[2]) < 1e-7: # Horizontal line
                     target_z = start_p_side[2]
                     target_x = far_x_pos_side if dir_s_side[0] > 1e-7 else (far_x_neg_side if dir_s_side[0] < -1e-7 else start_p_side[0])
                     end_points_side[i_side] = np.array([target_x, 0, target_z])
                else:
                     end_points_side[i_side] = _intersect_z0(start_p_side, dir_s_side)

            # Plot the lines
            label_circ_spread_side_vis = f'Разлёт от края ({full_angle_deg_for_label:.0f}°)'
            if end_points_side[0] is not None:
                ax_side.plot([start_edge1_circ_vis[0], end_points_side[0][0]], 
                             [start_edge1_circ_vis[2], end_points_side[0][2]],
                             color=emission_color, linestyle=emission_linestyle, 
                             alpha=circular_edge_spread_alpha, linewidth=circular_edge_spread_linewidth, 
                             label=label_circ_spread_side_vis)
                label_circ_spread_side_vis = None # Only add label once
            if end_points_side[1] is not None:
                ax_side.plot([start_edge2_circ_vis[0], end_points_side[1][0]], 
                             [start_edge2_circ_vis[2], end_points_side[1][2]],
                             color=emission_color, linestyle=emission_linestyle, 
                             alpha=circular_edge_spread_alpha, linewidth=circular_edge_spread_linewidth, 
                             label=label_circ_spread_side_vis)

        else: # Treat as point source for side view spread lines if radius is zero
            end_point_b1_z0_point = _intersect_z0(source_center_global, dir_spread1_global_cone_edge_xz)
            end_point_b2_z0_point = _intersect_z0(source_center_global, dir_spread2_global_cone_edge_xz)
            label_point_spread_side = f'Разлёт эмиссии ({full_angle_deg_for_label:.0f}°)'
            if end_point_b1_z0_point is not None:
                ax_side.plot([source_center_global[0], end_point_b1_z0_point[0]], 
                             [source_center_global[2], end_point_b1_z0_point[2]],
                             color=emission_color, linestyle=emission_linestyle, 
                             alpha=general_line_alpha, # Use general alpha for point-like spread
                             linewidth=common_data['linewidths'].get('spread_center', 1.5), # Use specific linewidth if defined
                             label=label_point_spread_side)
                label_point_spread_side = None
            if end_point_b2_z0_point is not None:
                ax_side.plot([source_center_global[0], end_point_b2_z0_point[0]], 
                             [source_center_global[2], end_point_b2_z0_point[2]],
                             color=emission_color, linestyle=emission_linestyle, 
                             alpha=general_line_alpha, 
                             linewidth=common_data['linewidths'].get('spread_center', 1.5), 
                             label=label_point_spread_side)

        # --- Вид сверху (Top View) ---
        # Calculate emission footprint points on Z=0
        all_intersection_xy_points = []
        source_sample_points_local_top = []
        if src_radius_circ > 1e-6:
            num_edge_samples_top = 36 # Sample points on the edge of the circular source
            theta_edge_top = np.linspace(0, 2*np.pi, num_edge_samples_top, endpoint=False)
            source_sample_points_local_top = [np.array([src_radius_circ * math.cos(t), src_radius_circ * math.sin(t), 0]) 
                                              for t in theta_edge_top]
            # Add center point for better coverage if source is large compared to spread
            # source_sample_points_local_top.append(np.zeros(3)) 
        else: 
            source_sample_points_local_top = [np.zeros(3)] # Treat as point source

        num_cone_boundary_rays_top = 24 # Rays per sample point
        
        draw_top_emission_polygon = True
        # Check if source is above Z=0 when angle is 90 deg
        max_src_z_abs = 0
        if source_sample_points_local_top:
             global_sample_points = _apply_rotation(np.array(source_sample_points_local_top).T, 
                                                   source_orient_matrix, source_center_global)
             if global_sample_points.ndim == 2 and global_sample_points.shape[0] == 3:
                  max_src_z_abs = np.max(np.abs(global_sample_points[2, :]))

        if is_angle_90_deg and max_src_z_abs > 1e-3: 
            draw_top_emission_polygon = False
            print("INFO (circular_emission): Emission angle ~90 deg and source is not on Z=0. Top emission footprint polygon not drawn.")

        if draw_top_emission_polygon:
            plot_xlim_top, plot_ylim_top = ax_top.get_xlim(), ax_top.get_ylim()
            span_x_top = abs(plot_xlim_top[1] - plot_xlim_top[0]) if abs(plot_xlim_top[1] - plot_xlim_top[0]) > 1e-3 else 1000.0
            span_y_top = abs(plot_ylim_top[1] - plot_ylim_top[0]) if abs(plot_ylim_top[1] - plot_ylim_top[0]) > 1e-3 else 1000.0
            far_dist_top = max(span_x_top, span_y_top) * 2.0

            # Rotation matrix to align local -Z with the global emission axis
            R_align_cone_axis = get_rotation_matrix_align_vectors(np.array([0,0,-1.0]), emission_axis_global)

            for p_local_top in source_sample_points_local_top: 
                start_point_global_on_src = _apply_rotation(p_local_top, source_orient_matrix, source_center_global) # Should be (3,)
                if start_point_global_on_src.shape != (3,): continue # Safety check

                phi_cone_rays_top = np.linspace(0, 2*np.pi, num_cone_boundary_rays_top, endpoint=False)
                for phi_cr_top in phi_cone_rays_top:
                    dir_local_cone_edge_top = np.array([math.sin(max_theta_rad) * math.cos(phi_cr_top),
                                                    math.sin(max_theta_rad) * math.sin(phi_cr_top),
                                                    -math.cos(max_theta_rad)]) # Shape (3,)
                    dir_global_cone_edge_top = R_align_cone_axis.dot(dir_local_cone_edge_top) # Shape (3,)
                    
                    if dir_global_cone_edge_top.shape != (3,): continue # Safety check

                    if is_angle_90_deg:
                        if abs(start_point_global_on_src[2]) < 1e-3 : 
                           xy_dir = dir_global_cone_edge_top[:2]
                           norm_xy_dir = np.linalg.norm(xy_dir)
                           if norm_xy_dir > 1e-7:
                               far_point = start_point_global_on_src[:2] + (xy_dir / norm_xy_dir) * far_dist_top
                               all_intersection_xy_points.append(far_point) # Append (x, y) array
                    else: 
                        intersect_pt_top = _intersect_z0(start_point_global_on_src, dir_global_cone_edge_top)
                        if intersect_pt_top is not None: # Shape (3,)
                            all_intersection_xy_points.append(intersect_pt_top[:2]) # Append (x, y) array

            # --- Process intersection points and draw ---
            emission_footprint_on_z0_vertices = []
            if all_intersection_xy_points:
                intersection_points_tuples = [tuple(p) for p in all_intersection_xy_points]
                if len(intersection_points_tuples) >= 3:
                    try:
                        emission_footprint_on_z0_vertices = _convex_hull(intersection_points_tuples)
                    except Exception as e_hull:
                        print(f"ERROR (circular_emission ConvexHull): {e_hull}. Plotting raw points.")
                        ax_top.plot(np.array(intersection_points_tuples)[:,0], 
                                    np.array(intersection_points_tuples)[:,1], 
                                    '.', color=emission_color, alpha=0.3, markersize=2, label='Точки эмиссии (ошибка hull)')
                        emission_footprint_on_z0_vertices = [] 

            # Get substrate boundary
            substrate_vertices_xy = _generate_substrate_boundary(params.get('target_type', config.TARGET_DISK), params) 
            
            # Calculate intersection polygon
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
                else: # No overlap found, clear the list
                     final_intersection_polygon_vertices = []

            # Draw the final intersection polygon
            if final_intersection_polygon_vertices and len(final_intersection_polygon_vertices) >=3:
                intersection_patch_top = patches.Polygon(
                    np.array(final_intersection_polygon_vertices), 
                    facecolor=emission_fill_color, edgecolor=emission_color,
                    alpha=emission_area_alpha, linewidth=polygon_outline_linewidth,
                    linestyle=emission_linestyle, zorder=5, label='Область эмиссии (Z=0)')
                ax_top.add_patch(intersection_patch_top)
            elif not final_intersection_polygon_vertices and emission_footprint_on_z0_vertices:
                 # Case where emission footprint exists but doesn't overlap substrate
                 pass # Don't draw anything if no intersection


    except Exception as e:
        print(f"ERROR in plot_circular_emission: {e}")
        traceback.print_exc()

