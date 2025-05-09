# coating_simulator_project/coating_simulator/visualization/geom_emission_ring.py
"""
Functions for plotting emission details specifically for a RING source.
Calculates intersection of emission footprint with substrate before drawing.
Uses the background color fill method to simulate a donut hole when applicable.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback

# Используем относительные импорты
from .geom_emission import _intersect_z0, get_rotation_matrix_align_vectors 
from .geom_utils import _convex_hull, _is_point_in_polygon, _polygon_area, _generate_substrate_boundary # Import necessary utils
from .geom_source import _apply_rotation 
from .. import config 

def plot_ring_emission(ax_top: plt.Axes, ax_side: plt.Axes, params: dict, common_data: dict):
    """
    Plots emission details for a ring source, clipping the footprint to the substrate.
    Отображает детали эмиссии для кольцевого источника, обрезая область по подложке.
    """
    try:
        # ... (Начало функции: получение параметров, расчет фокуса - без изменений) ...
        source_center_global = common_data['source_center_global'] 
        source_orient_matrix = common_data['source_orient_matrix'] 
        emission_axis_global = common_data['emission_axis_global'] 
        max_theta_rad = common_data['max_theta_rad']
        full_angle_deg_for_label = common_data['full_angle_deg_for_label']
        target_type = params.get('target_type', config.TARGET_DISK) # Get target type
        
        if source_center_global.shape != (3,): source_center_global = source_center_global.flatten()
        if emission_axis_global.shape != (3,): emission_axis_global = emission_axis_global.flatten()
        if source_center_global.shape != (3,) or emission_axis_global.shape != (3,):
             print("ERROR (ring_emission): Invalid shape for common_data vectors.")
             return

        emission_color = common_data['colors'].get('emission', 'orange')
        emission_fill_color = common_data['colors'].get('emission_fill', emission_color)
        focus_color = common_data['colors'].get('focus', 'purple')
        background_color = ax_top.get_facecolor() 
        if background_color == 'none' or not isinstance(background_color, (str, tuple)):
             background_color = 'white' 
        emission_area_alpha = common_data['alphas'].get('emission_area_fill', 0.4)
        general_line_alpha = common_data['alphas'].get('general_line', 0.75) 
        polygon_outline_linewidth = common_data['linewidths'].get('polygon_outline', 1.2)
        ring_edge_spread_linewidth = common_data['linewidths'].get('ring_edge_spread', 1.0)
        focus_line_style = ':' 
        emission_linestyle = '--' 

        src_diameter_ring = params.get('src_diameter', 10.0)
        src_radius_ring = src_diameter_ring / 2.0
        focus_point_param_ring = params.get('focus_point', config.INFINITY_SYMBOL)

        is_angle_90_deg = abs(max_theta_rad - math.pi / 2.0) < 1e-7

        focus_point_global_ring = None 
        is_parallel_focus_ring = False 
        # ... (focus calculation remains the same) ...
        if focus_point_param_ring == config.INFINITY_SYMBOL:
            is_parallel_focus_ring = True
        else:
            try:
                focus_L_ring = float(focus_point_param_ring) 
                focus_point_global_ring = source_center_global + emission_axis_global * focus_L_ring 
                if focus_point_global_ring is not None:
                    if not ax_top.findobj(match=lambda x: x.get_label() == 'Точка фокуса L'):
                        ax_top.plot(focus_point_global_ring[0], focus_point_global_ring[1], 'x', color=focus_color, markersize=7, label='Точка фокуса L', zorder=10) 
                    if not ax_side.findobj(match=lambda x: x.get_label() == 'Точка фокуса L'):
                        ax_side.plot(focus_point_global_ring[0], focus_point_global_ring[2], 'x', color=focus_color, markersize=7, label='Точка фокуса L', zorder=10)
            except ValueError:
                is_parallel_focus_ring = True 

        if src_radius_ring <= 1e-6: 
            print("INFO (ring_emission): Ring radius is very small, plotting as point source emission.")
            return

        # --- Вид сбоку (Side View) ---
        # (Side view logic remains unchanged)
        # ... (copy side view logic here) ...
        edge_points_local_side_view_ring = [np.array([src_radius_ring, 0, 0]), np.array([-src_radius_ring, 0, 0])]
        p1_y_edge_g = _apply_rotation(np.array([0, src_radius_ring, 0]), source_orient_matrix, source_center_global) 
        p2_y_edge_g = _apply_rotation(np.array([0, -src_radius_ring, 0]), source_orient_matrix, source_center_global) 
        p1_x_edge_g = _apply_rotation(np.array([src_radius_ring, 0, 0]), source_orient_matrix, source_center_global) 
        p2_x_edge_g = _apply_rotation(np.array([-src_radius_ring, 0, 0]), source_orient_matrix, source_center_global) 
        
        if abs(p1_y_edge_g[0] - p2_y_edge_g[0]) > abs(p1_x_edge_g[0] - p2_x_edge_g[0]):
            edge_points_local_side_view_ring = [np.array([0, src_radius_ring, 0]), np.array([0, -src_radius_ring, 0])]
        
        start_points_side_view_ring = [_apply_rotation(p_loc, source_orient_matrix, source_center_global) 
                                       for p_loc in edge_points_local_side_view_ring] 
        
        focus_label_applied_ring = False
        emission_spread_label_applied_ring = False
        xlim_side_plot = ax_side.get_xlim()
        plot_span_x_side = abs(xlim_side_plot[1] - xlim_side_plot[0])
        far_x_neg_side = xlim_side_plot[0] - 0.1 * plot_span_x_side 
        far_x_pos_side = xlim_side_plot[1] + 0.1 * plot_span_x_side

        for start_p_global_ring in start_points_side_view_ring: 
            focusing_direction_global_from_point = emission_axis_global.copy() 
            if not is_parallel_focus_ring and focus_point_global_ring is not None:
                direction_to_focus = (focus_point_global_ring - start_p_global_ring).flatten() 
                if direction_to_focus.shape == (3,): 
                    dist_to_focus = np.linalg.norm(direction_to_focus)
                    if dist_to_focus > 1e-6:
                        focusing_direction_global_from_point = direction_to_focus / dist_to_focus 
                else:
                     focusing_direction_global_from_point = emission_axis_global.copy() 

            if focusing_direction_global_from_point.shape != (3,): continue 

            end_point_focus_z0_ring = _intersect_z0(start_p_global_ring, focusing_direction_global_from_point)
            if end_point_focus_z0_ring is not None: 
                label_f_ring = 'Фокусировка (от края)' if not focus_label_applied_ring else None 
                ax_side.plot([start_p_global_ring[0], end_point_focus_z0_ring[0]], 
                             [start_p_global_ring[2], end_point_focus_z0_ring[2]],
                             color=focus_color, linestyle=focus_line_style, alpha=general_line_alpha, label=label_f_ring) 
                if label_f_ring: focus_label_applied_ring = True

            focus_dir_z = focusing_direction_global_from_point[2]
            focus_dir_x = focusing_direction_global_from_point[0]
            angle_of_focusing_line_xz = math.atan2(focus_dir_x, -focus_dir_z) if abs(focus_dir_z) > 1e-7 else \
                                         (math.pi/2 if focus_dir_x > 0 else -math.pi/2)

            dir_spread1_xz_view = np.array([math.sin(angle_of_focusing_line_xz - max_theta_rad), 0, 
                                            -math.cos(angle_of_focusing_line_xz - max_theta_rad)])
            dir_spread2_xz_view = np.array([math.sin(angle_of_focusing_line_xz + max_theta_rad), 0, 
                                            -math.cos(angle_of_focusing_line_xz + max_theta_rad)])
            
            end_points_s_z0_ring = [None, None]
            spread_dirs_xz = [dir_spread1_xz_view, dir_spread2_xz_view]

            for i_spread, dir_s_xz in enumerate(spread_dirs_xz): 
                if abs(dir_s_xz[2]) < 1e-7: 
                    target_z = start_p_global_ring[2] 
                    target_x = far_x_pos_side if dir_s_xz[0] > 1e-7 else (far_x_neg_side if dir_s_xz[0] < -1e-7 else start_p_global_ring[0])
                    end_points_s_z0_ring[i_spread] = np.array([target_x, 0, target_z]) 
                else: 
                    end_points_s_z0_ring[i_spread] = _intersect_z0(start_p_global_ring, dir_s_xz) 

            label_e_ring = f'Разлёт от края ({full_angle_deg_for_label:.0f}°)' if not emission_spread_label_applied_ring else None
            if end_points_s_z0_ring[0] is not None: 
                ax_side.plot([start_p_global_ring[0], end_points_s_z0_ring[0][0]], 
                             [start_p_global_ring[2], end_points_s_z0_ring[0][2]], 
                             color=emission_color, linestyle=emission_linestyle, 
                             linewidth=ring_edge_spread_linewidth, alpha=general_line_alpha, 
                             label=label_e_ring)
                if label_e_ring: emission_spread_label_applied_ring = True
            
            current_label_for_s2_ring = label_e_ring if not emission_spread_label_applied_ring and label_e_ring else None
            if end_points_s_z0_ring[1] is not None: 
                ax_side.plot([start_p_global_ring[0], end_points_s_z0_ring[1][0]], 
                             [start_p_global_ring[2], end_points_s_z0_ring[1][2]],
                             color=emission_color, linestyle=emission_linestyle, 
                             linewidth=ring_edge_spread_linewidth, alpha=general_line_alpha, 
                             label=current_label_for_s2_ring)
                if current_label_for_s2_ring : emission_spread_label_applied_ring = True
        
        # --- Вид сверху (Top View) ---
        num_points_on_ring_for_top = 36 
        theta_ring_top = np.linspace(0, 2 * np.pi, num_points_on_ring_for_top, endpoint=False)
        x_ring_local_top = src_radius_ring * np.cos(theta_ring_top)
        y_ring_local_top = src_radius_ring * np.sin(theta_ring_top)
        z_ring_local_top = np.zeros_like(theta_ring_top) 
        
        points_local_ring_top = np.vstack((x_ring_local_top, y_ring_local_top, z_ring_local_top))
        ring_points_global_top = _apply_rotation(points_local_ring_top, source_orient_matrix, source_center_global) # Shape (3, N)

        # --- Calculate intersection points for INNER and OUTER boundaries ---
        all_inner_intersection_xy = [] 
        all_outer_intersection_xy = [] 
        num_cone_boundary_rays_top_ring = 16 

        draw_top_emission_polygon = True
        max_ring_z_abs = np.max(np.abs(ring_points_global_top[2, :])) if ring_points_global_top.shape[1] > 0 else 0
        if is_angle_90_deg and max_ring_z_abs > 1e-3: 
            draw_top_emission_polygon = False
            print("INFO (ring_emission): Emission angle ~90 deg and ring source is not on Z=0. Top emission footprint polygon not drawn.")

        if draw_top_emission_polygon:
            plot_xlim_top, plot_ylim_top = ax_top.get_xlim(), ax_top.get_ylim()
            span_x_top = abs(plot_xlim_top[1] - plot_xlim_top[0]) if abs(plot_xlim_top[1] - plot_xlim_top[0]) > 1e-3 else 1000.0
            span_y_top = abs(plot_ylim_top[1] - plot_ylim_top[0]) if abs(plot_ylim_top[1] - plot_ylim_top[0]) > 1e-3 else 1000.0
            far_dist_top = max(span_x_top, span_y_top) * 2.0

            for i in range(num_points_on_ring_for_top): 
                current_ring_point_global_spread = ring_points_global_top[:, i] # Shape (3,)
                
                axis_of_emission_cone_from_pt = emission_axis_global.copy() # Shape (3,)
                if not is_parallel_focus_ring and focus_point_global_ring is not None:
                    direction_to_focus_from_pt = (focus_point_global_ring - current_ring_point_global_spread).flatten() # Ensure (3,)
                    if direction_to_focus_from_pt.shape == (3,):
                        dist_to_focus_from_pt = np.linalg.norm(direction_to_focus_from_pt)
                        if dist_to_focus_from_pt > 1e-6:
                            axis_of_emission_cone_from_pt = direction_to_focus_from_pt / dist_to_focus_from_pt # Shape (3,)
                    else:
                         axis_of_emission_cone_from_pt = emission_axis_global.copy()

                if axis_of_emission_cone_from_pt.shape != (3,): continue 

                R_align_cone_axis = get_rotation_matrix_align_vectors(np.array([0,0,-1.0]), axis_of_emission_cone_from_pt) 
                
                phi_cone_rays = np.linspace(0, 2*np.pi, num_cone_boundary_rays_top_ring, endpoint=False)
                for phi_cr in phi_cone_rays: 
                    # Outer boundary ray (+max_theta_rad)
                    dir_local_outer = np.array([math.sin(max_theta_rad) * math.cos(phi_cr),
                                                math.sin(max_theta_rad) * math.sin(phi_cr),
                                                -math.cos(max_theta_rad)]) 
                    spread_ray_global_outer = R_align_cone_axis.dot(dir_local_outer) 
                    
                    # Inner boundary ray (-max_theta_rad relative to axis, same phi)
                    dir_local_inner = np.array([math.sin(-max_theta_rad) * math.cos(phi_cr),
                                                math.sin(-max_theta_rad) * math.sin(phi_cr),
                                                -math.cos(max_theta_rad)]) 
                    spread_ray_global_inner = R_align_cone_axis.dot(dir_local_inner) 
                    
                    if spread_ray_global_outer.shape != (3,) or spread_ray_global_inner.shape != (3,): continue

                    # Calculate intersections
                    if is_angle_90_deg: 
                        if abs(current_ring_point_global_spread[2]) < 1e-3 : 
                           # Outer
                           xy_dir_outer = spread_ray_global_outer[:2]
                           norm_xy_outer = np.linalg.norm(xy_dir_outer)
                           if norm_xy_outer > 1e-7:
                               far_point_outer = current_ring_point_global_spread[:2] + (xy_dir_outer / norm_xy_outer) * far_dist_top
                               all_outer_intersection_xy.append(far_point_outer) 
                           # Inner 
                           xy_dir_inner = spread_ray_global_inner[:2]
                           norm_xy_inner = np.linalg.norm(xy_dir_inner)
                           if norm_xy_inner > 1e-7:
                               far_point_inner = current_ring_point_global_spread[:2] + (xy_dir_inner / norm_xy_inner) * far_dist_top
                               all_inner_intersection_xy.append(far_point_inner) 
                    else: # Angle < 90 deg
                        intersect_pt_outer = _intersect_z0(current_ring_point_global_spread, spread_ray_global_outer)
                        if intersect_pt_outer is not None: 
                            all_outer_intersection_xy.append(intersect_pt_outer[:2]) 
                        
                        intersect_pt_inner = _intersect_z0(current_ring_point_global_spread, spread_ray_global_inner)
                        if intersect_pt_inner is not None: 
                            all_inner_intersection_xy.append(intersect_pt_inner[:2]) 

            # --- Calculate Hulls ---
            outer_hull_vertices_tuples = []
            inner_hull_vertices_tuples = []
            
            if all_outer_intersection_xy and len(all_outer_intersection_xy) >= 3:
                try:
                    outer_hull_vertices_tuples = _convex_hull([tuple(p) for p in all_outer_intersection_xy])
                except Exception as e_hull_out:
                    print(f"ERROR (ring_emission Outer ConvexHull): {e_hull_out}.") 
                    outer_hull_vertices_tuples = [] 
            
            if all_inner_intersection_xy and len(all_inner_intersection_xy) >= 3:
                try:
                    inner_hull_vertices_tuples = _convex_hull([tuple(p) for p in all_inner_intersection_xy])
                except Exception as e_hull_in:
                    print(f"ERROR (ring_emission Inner ConvexHull): {e_hull_in}.")
                    inner_hull_vertices_tuples = [] 

            # --- Get Substrate Boundary ---
            substrate_vertices_xy = _generate_substrate_boundary(target_type, params) 

            # --- Calculate Intersection Polygons ---
            final_outer_intersection_vertices = []
            final_inner_intersection_vertices = [] # For the hole

            if outer_hull_vertices_tuples and substrate_vertices_xy:
                intersection_points_outer = []
                # Points from outer hull inside substrate
                for pt_emission in outer_hull_vertices_tuples:
                    if _is_point_in_polygon(pt_emission, substrate_vertices_xy):
                        intersection_points_outer.append(pt_emission)
                # Points from substrate inside outer hull
                for pt_substrate in substrate_vertices_xy:
                    if _is_point_in_polygon(pt_substrate, outer_hull_vertices_tuples):
                        intersection_points_outer.append(pt_substrate)
                # Compute final outer intersection hull
                if intersection_points_outer and len(intersection_points_outer) >= 3:
                     try:
                         final_outer_intersection_vertices = _convex_hull(intersection_points_outer)
                     except Exception as e_fout_hull:
                          print(f"ERROR (ring_emission Final Outer Hull): {e_fout_hull}")
                          final_outer_intersection_vertices = []

            if inner_hull_vertices_tuples and substrate_vertices_xy:
                intersection_points_inner = []
                # Points from inner hull inside substrate
                for pt_emission in inner_hull_vertices_tuples:
                    if _is_point_in_polygon(pt_emission, substrate_vertices_xy):
                        intersection_points_inner.append(pt_emission)
                # Points from substrate inside inner hull
                for pt_substrate in substrate_vertices_xy:
                    if _is_point_in_polygon(pt_substrate, inner_hull_vertices_tuples):
                        intersection_points_inner.append(pt_substrate)
                # Compute final inner intersection hull
                if intersection_points_inner and len(intersection_points_inner) >= 3:
                     try:
                         final_inner_intersection_vertices = _convex_hull(intersection_points_inner)
                     except Exception as e_fin_hull:
                          print(f"ERROR (ring_emission Final Inner Hull): {e_fin_hull}")
                          final_inner_intersection_vertices = []


            # --- Decide whether to draw donut based on INTERSECTION polygons ---
            draw_donut = False
            if final_outer_intersection_vertices and len(final_outer_intersection_vertices) >= 3 and \
               final_inner_intersection_vertices and len(final_inner_intersection_vertices) >= 3:
                outer_intersect_area = _polygon_area(final_outer_intersection_vertices)
                inner_intersect_area = _polygon_area(final_inner_intersection_vertices)
                if inner_intersect_area > 1e-6 and outer_intersect_area > inner_intersect_area + 1e-6:
                    inner_centroid = np.mean(np.array(final_inner_intersection_vertices), axis=0)
                    if _is_point_in_polygon(tuple(inner_centroid), final_outer_intersection_vertices):
                        draw_donut = True
                    else:
                        print("INFO (ring_emission): Inner INTERSECTION hull centroid outside outer INTERSECTION hull.")
            
            print(f"DEBUG (ring_emission): Final Outer Intersection pts: {len(final_outer_intersection_vertices)}, "
                  f"Final Inner Intersection pts: {len(final_inner_intersection_vertices)}, Draw Donut: {draw_donut}")

            # --- Actual Drawing using clipped polygons ---
            if final_outer_intersection_vertices and len(final_outer_intersection_vertices) >= 3:
                # 1. Draw outer intersection polygon filled
                polygon_outer_fill = patches.Polygon(
                    np.array(final_outer_intersection_vertices), 
                    closed=True, edgecolor='none', facecolor=emission_fill_color, 
                    alpha=emission_area_alpha, zorder=4)
                ax_top.add_patch(polygon_outer_fill)

                # 2. If donut, draw inner intersection polygon with background
                if draw_donut and final_inner_intersection_vertices and len(final_inner_intersection_vertices) >= 3:
                    polygon_inner_hole = patches.Polygon(
                        np.array(final_inner_intersection_vertices), closed=True,
                        edgecolor='none', facecolor=background_color, alpha=1.0, zorder=5)
                    ax_top.add_patch(polygon_inner_hole)

                # 3. Draw outlines for the intersection polygons
                outer_outline = np.array(final_outer_intersection_vertices + [final_outer_intersection_vertices[0]]) 
                label_emission = 'Область эмиссии (Z=0)' if not ax_top.findobj(match=lambda x: x.get_label() == 'Область эмиссии (Z=0)') else None
                ax_top.plot(outer_outline[:, 0], outer_outline[:, 1],
                            color=emission_color, linewidth=polygon_outline_linewidth, 
                            linestyle=emission_linestyle, alpha=general_line_alpha, 
                            zorder=7, label=label_emission) 
                
                if draw_donut and final_inner_intersection_vertices and len(final_inner_intersection_vertices) >= 3:
                    inner_outline = np.array(final_inner_intersection_vertices + [final_inner_intersection_vertices[0]]) 
                    ax_top.plot(inner_outline[:, 0], inner_outline[:, 1],
                                color=emission_color, linewidth=polygon_outline_linewidth * 0.8, 
                                linestyle=emission_linestyle, alpha=general_line_alpha, zorder=7) 
            
            elif all_outer_intersection_xy: # Fallback if outer hull failed or no intersection
                 # Check if there were original points, even if no intersection
                 if not final_outer_intersection_vertices:
                      print("INFO (ring_emission): Emission footprint does not intersect substrate.")
                 # Optionally plot raw points if needed for debugging
                 # points_np = np.array([tuple(p) for p in all_outer_intersection_xy])
                 # ax_top.plot(points_np[:,0], points_np[:,1], 
                 #             '.', color=emission_color, alpha=0.1, markersize=1)


        # --- Draw the focus line polygon (if calculated) ---
        # (Focus line logic remains unchanged, still drawn based on full focus hull)
        intersect_points_xy_focus_ring = [] 
        if not is_angle_90_deg: 
            # ... (focus point calculation) ...
            for i in range(num_points_on_ring_for_top):
                current_ring_point_global = ring_points_global_top[:, i] 
                focusing_dir_from_curr_point_top = emission_axis_global.copy() 
                if not is_parallel_focus_ring and focus_point_global_ring is not None:
                    direction_top = (focus_point_global_ring - current_ring_point_global).flatten() 
                    if direction_top.shape == (3,):
                        dist_top = np.linalg.norm(direction_top)
                        if dist_top > 1e-6: focusing_dir_from_curr_point_top = direction_top / dist_top 
                    else:
                         focusing_dir_from_curr_point_top = emission_axis_global.copy()
                
                if focusing_dir_from_curr_point_top.shape != (3,): continue
                
                end_point_z0_focus_top = _intersect_z0(current_ring_point_global, focusing_dir_from_curr_point_top)
                if end_point_z0_focus_top is not None: 
                    intersect_points_xy_focus_ring.append(end_point_z0_focus_top[:2])

        if intersect_points_xy_focus_ring and len(intersect_points_xy_focus_ring) >= 3:
             focus_points_tuples = [tuple(p) for p in intersect_points_xy_focus_ring]
             try:
                focus_hull_vertices_tuples = _convex_hull(focus_points_tuples)
                # Draw focus line clipped to substrate? More complex. Draw full line for now.
                if focus_hull_vertices_tuples and len(focus_hull_vertices_tuples) >= 3:
                    polygon_focus_line = patches.Polygon(
                        np.array(focus_hull_vertices_tuples), 
                        closed=True, 
                        edgecolor=focus_color, facecolor='none', 
                        linewidth=polygon_outline_linewidth, linestyle=focus_line_style,      
                        alpha=general_line_alpha, zorder=8, # Higher zorder for focus line
                        label='Линия макс. интенсивности (Z=0)' )
                    ax_top.add_patch(polygon_focus_line)
                elif focus_points_tuples: 
                     points_np = np.array(focus_points_tuples)
                     ax_top.plot(points_np[:,0], points_np[:,1], 
                                 '.', color=focus_color, alpha=0.5, markersize=3, zorder=8, label='Точки макс. интенсивности (Z=0)')

             except Exception as e_focus_poly:
                 print(f"Could not draw focus intensity line for ring source: {e_focus_poly}")
                 traceback.print_exc()
                 points_np = np.array(focus_points_tuples) 
                 ax_top.plot(points_np[:,0], points_np[:,1], 
                             '.', color=focus_color, alpha=0.5, markersize=3, zorder=8, label='Точки макс. интенсивности (ошибка)')
        elif intersect_points_xy_focus_ring: 
            points_np = np.array([tuple(p) for p in intersect_points_xy_focus_ring])
            ax_top.plot(points_np[:,0], points_np[:,1], 
                        '.', color=focus_color, alpha=0.5, markersize=3, zorder=8, label='Точки макс. интенсивности (мало точек)')


    except Exception as e:
        print(f"ERROR in plot_ring_emission: {e}")
        traceback.print_exc()
