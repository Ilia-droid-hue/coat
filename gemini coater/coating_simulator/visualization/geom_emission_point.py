# coating_simulator_project/coating_simulator/visualization/geom_emission_point.py
"""
Functions for plotting emission details specifically for a POINT source.
Clips the emission footprint to the substrate boundary.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback 

# Используем относительные импорты
from .geom_emission import _intersect_z0, get_rotation_matrix_align_vectors # Helpers from geom_emission
from .geom_utils import _convex_hull, _is_point_in_polygon, _generate_substrate_boundary # Helpers from geom_utils
from .geom_source import _apply_rotation # Should return (3,) for single point
from .. import config

def plot_point_emission(ax_top: plt.Axes, ax_side: plt.Axes, params: dict, common_data: dict):
    """
    Plots emission for a point source, clipped to the substrate boundary.
    Side view: Central dashed axis and two spread lines from the source center.
    Top view: Filled outline of the emission intersection area with dashed border.
    Отображает эмиссию для точечного источника, обрезанную по границе подложки.
    Вид сбоку: Центральная пунктирная ось и две линии разлета из центра источника.
    Вид сверху: Залитый контур области пересечения эмиссии с пунктирной границей.
    """
    try:
        source_center_global = common_data['source_center_global']
        emission_axis_global = common_data['emission_axis_global']
        max_theta_rad = common_data['max_theta_rad']
        full_angle_deg_for_label = common_data['full_angle_deg_for_label']
        target_type = params.get('target_type', config.TARGET_DISK) # Get target type

        # Ensure vectors are 1D
        if source_center_global.shape != (3,): source_center_global = source_center_global.flatten()
        if emission_axis_global.shape != (3,): emission_axis_global = emission_axis_global.flatten()
        if source_center_global.shape != (3,) or emission_axis_global.shape != (3,):
             print("ERROR (point_emission): Invalid shape for common_data vectors.")
             return
        
        # Styling
        emission_color = common_data['colors'].get('emission', 'orange') 
        emission_fill_color = common_data['colors'].get('emission_fill', emission_color) 
        general_line_alpha = common_data['alphas'].get('general_line', 0.75) 
        emission_area_alpha = common_data['alphas'].get('emission_area_fill', 0.4) 
        source_orient_matrix = common_data['source_orient_matrix'] # Needed for ray directions
        spread_center_linewidth = common_data['linewidths'].get('spread_center', 1.5)
        polygon_outline_linewidth = common_data['linewidths'].get('polygon_outline', 1.2) 
        emission_linestyle = '--'

        is_angle_90_deg = abs(max_theta_rad - math.pi / 2.0) < 1e-7

        # --- Вид сбоку (Side View) ---
        # (Side view logic remains the same)
        angle_of_main_axis_xz = math.atan2(emission_axis_global[0], -emission_axis_global[2]) \
                                  if abs(emission_axis_global[2]) > 1e-7 else \
                                  (math.pi/2 if emission_axis_global[0] > 0 else -math.pi/2)
        
        dir_spread1_global_cone_edge_xz = np.array([math.sin(angle_of_main_axis_xz - max_theta_rad), 0, 
                                                 -math.cos(angle_of_main_axis_xz - max_theta_rad)])
        dir_spread2_global_cone_edge_xz = np.array([math.sin(angle_of_main_axis_xz + max_theta_rad), 0, 
                                                 -math.cos(angle_of_main_axis_xz + max_theta_rad)])

        # Handle 90 degree case for side view lines
        xlim_side_plot = ax_side.get_xlim()
        plot_span_x_side = abs(xlim_side_plot[1] - xlim_side_plot[0])
        far_x_neg_side = xlim_side_plot[0] - 0.1 * plot_span_x_side 
        far_x_pos_side = xlim_side_plot[1] + 0.1 * plot_span_x_side

        end_points_side = [None, None]
        spread_dirs_side = [dir_spread1_global_cone_edge_xz, dir_spread2_global_cone_edge_xz]

        for i_side, dir_s_side in enumerate(spread_dirs_side):
            if abs(dir_s_side[2]) < 1e-7: # Horizontal line
                 target_z = source_center_global[2]
                 target_x = far_x_pos_side if dir_s_side[0] > 1e-7 else (far_x_neg_side if dir_s_side[0] < -1e-7 else source_center_global[0])
                 end_points_side[i_side] = np.array([target_x, 0, target_z])
            else:
                 end_points_side[i_side] = _intersect_z0(source_center_global, dir_s_side)

        label_spread_side = f'Разлёт эмиссии ({full_angle_deg_for_label:.0f}°)'
        if end_points_side[0] is not None:
            ax_side.plot([source_center_global[0], end_points_side[0][0]], 
                         [source_center_global[2], end_points_side[0][2]],
                         color=emission_color, linestyle=emission_linestyle, # Use emission linestyle
                         alpha=general_line_alpha, 
                         linewidth=spread_center_linewidth, label=label_spread_side)
            label_spread_side = None 
        if end_points_side[1] is not None:
            ax_side.plot([source_center_global[0], end_points_side[1][0]], 
                         [source_center_global[2], end_points_side[1][2]],
                         color=emission_color, linestyle=emission_linestyle, # Use emission linestyle
                         alpha=general_line_alpha, 
                         linewidth=spread_center_linewidth, label=label_spread_side)

        # --- Вид сверху (Top View) ---
        all_intersection_xy_points = []
        num_cone_boundary_rays_top = 32 
        
        draw_top_emission_polygon = True
        if is_angle_90_deg and abs(source_center_global[2]) > 1e-3: 
            draw_top_emission_polygon = False
            print("INFO (point_emission): Emission angle ~90 deg and source is not on Z=0. Top emission footprint polygon not drawn.")

        if draw_top_emission_polygon:
            plot_xlim_top, plot_ylim_top = ax_top.get_xlim(), ax_top.get_ylim()
            span_x_top = abs(plot_xlim_top[1] - plot_xlim_top[0]) if abs(plot_xlim_top[1] - plot_xlim_top[0]) > 1e-3 else 1000.0
            span_y_top = abs(plot_ylim_top[1] - plot_ylim_top[0]) if abs(plot_ylim_top[1] - plot_ylim_top[0]) > 1e-3 else 1000.0
            far_dist_top = max(span_x_top, span_y_top) * 2.0

            # Rotation matrix to align local -Z axis with the global emission axis
            R_align_cone_axis = get_rotation_matrix_align_vectors(np.array([0,0,-1.0]), emission_axis_global)

            phi_cone_rays_top = np.linspace(0, 2*np.pi, num_cone_boundary_rays_top, endpoint=False)
            for phi_cr_top in phi_cone_rays_top:
                dir_local_cone_edge_top = np.array([math.sin(max_theta_rad) * math.cos(phi_cr_top),
                                                math.sin(max_theta_rad) * math.sin(phi_cr_top),
                                                -math.cos(max_theta_rad)]) # Shape (3,)
                dir_global_cone_edge_top = R_align_cone_axis.dot(dir_local_cone_edge_top) # Shape (3,)
                
                if dir_global_cone_edge_top.shape != (3,): continue # Safety check

                if is_angle_90_deg:
                    if abs(source_center_global[2]) < 1e-3 : 
                       xy_dir = dir_global_cone_edge_top[:2]
                       norm_xy_dir = np.linalg.norm(xy_dir)
                       if norm_xy_dir > 1e-7:
                           far_point = source_center_global[:2] + (xy_dir / norm_xy_dir) * far_dist_top
                           all_intersection_xy_points.append(far_point) # Append (x, y) array
                else: 
                    intersect_pt_top = _intersect_z0(source_center_global, dir_global_cone_edge_top)
                    if intersect_pt_top is not None: # Shape (3,)
                        all_intersection_xy_points.append(intersect_pt_top[:2]) # Append (x, y) array

            # --- Calculate Emission Footprint Hull ---
            emission_footprint_on_z0_vertices = []
            if all_intersection_xy_points and len(all_intersection_xy_points) >= 3:
                try:
                    emission_footprint_on_z0_vertices = _convex_hull([tuple(p) for p in all_intersection_xy_points])
                except Exception as e_hull:
                    print(f"ERROR (point_emission ConvexHull): {e_hull}. Plotting raw points.")
                    ax_top.plot(np.array(all_intersection_xy_points)[:,0], 
                                np.array(all_intersection_xy_points)[:,1], 
                                '.', color=emission_color, alpha=0.3, markersize=2, label='Точки эмиссии (ошибка hull)')
                    emission_footprint_on_z0_vertices = [] 

            # --- Get Substrate Boundary ---
            substrate_vertices_xy = _generate_substrate_boundary(target_type, params) 

            # --- Calculate Intersection Polygon ---
            final_intersection_polygon_vertices = [] 
            if emission_footprint_on_z0_vertices and substrate_vertices_xy:
                intersection_points = []
                # Points from emission hull inside substrate
                for pt_emission in emission_footprint_on_z0_vertices:
                    if _is_point_in_polygon(pt_emission, substrate_vertices_xy):
                        intersection_points.append(pt_emission)
                # Points from substrate inside emission hull
                for pt_substrate in substrate_vertices_xy:
                    if _is_point_in_polygon(pt_substrate, emission_footprint_on_z0_vertices):
                        intersection_points.append(pt_substrate)
                # Compute final intersection hull
                if intersection_points and len(intersection_points) >= 3:
                     try:
                         final_intersection_polygon_vertices = _convex_hull(intersection_points)
                     except Exception as e_f_hull:
                          print(f"ERROR (point_emission Final Hull): {e_f_hull}")
                          final_intersection_polygon_vertices = []

            # --- Draw the final intersection polygon ---
            if final_intersection_polygon_vertices and len(final_intersection_polygon_vertices) >= 3:
                intersection_patch_top = patches.Polygon(
                    np.array(final_intersection_polygon_vertices), 
                    closed=True, 
                    edgecolor=emission_color,       # Use emission color for edge
                    facecolor=emission_fill_color,  # Use fill color
                    linewidth=polygon_outline_linewidth, 
                    linestyle=emission_linestyle,   # Use dashed style
                    alpha=emission_area_alpha, 
                    zorder=5, 
                    label='Область эмиссии (Z=0)' 
                )
                ax_top.add_patch(intersection_patch_top)
            elif not final_intersection_polygon_vertices and emission_footprint_on_z0_vertices:
                 # Case where emission footprint exists but doesn't overlap substrate
                 print("INFO (point_emission): Emission footprint does not intersect substrate.")
                 # Optionally draw the full footprint outline if desired
                 # full_footprint_patch = patches.Polygon(
                 #      np.array(emission_footprint_on_z0_vertices), facecolor='none', 
                 #      edgecolor=emission_color, alpha=0.3, linewidth=0.8, linestyle=':')
                 # ax_top.add_patch(full_footprint_patch)
            elif all_intersection_xy_points: # Fallback if hull failed or no intersection
                 points_np = np.array([tuple(p) for p in all_intersection_xy_points])
                 ax_top.plot(points_np[:,0], points_np[:,1], 
                             '.', color=emission_color, alpha=0.3, markersize=2, label='Точки эмиссии (мало/ошибка)')

    except Exception as e:
        print(f"ERROR in plot_point_emission: {e}")
        traceback.print_exc()

