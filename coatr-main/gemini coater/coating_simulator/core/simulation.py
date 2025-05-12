# coating_simulator_project/coating_simulator/core/simulation.py
"""
Core simulation functions for different target types using Monte Carlo method.
Основные функции симуляции для различных типов мишеней методом Монте-Карло.
Optimized with analytical ray-target intersection and multiprocessing.
Worker further optimized with vectorized particle processing.
Grid setup for linear target now correctly uses length for X and width for Y.
Linear target movement path updated for symmetry around the source's X-position,
with overruns based on the source's X-offset from GUI.
Movement resolution within worker sub-chunks improved by using mini-batches,
with mini-batch size configurable from GUI.
Ensure mini_batch_size is passed for all simulation types.
Corrected grid cell definition and indexing for coverage map.
"""

import numpy as np
import math
import time
import os
import multiprocessing

# Используем относительные импорты из текущего пакета .core
from .distribution import rotation_matrix, sample_source_position_vectorized, sample_emission_vector_vectorized
from .. import config

# --- Вспомогательные функции для аналитического пересечения (без изменений) ---
def _intersect_ray_plane_analytical(ray_origin: np.ndarray, ray_direction: np.ndarray,
                                   plane_point: np.ndarray = np.array([0,0,0]),
                                   plane_normal: np.ndarray = np.array([0,0,1])) -> tuple[np.ndarray | None, float | None]:
    ray_origin = np.asarray(ray_origin, dtype=float)
    ray_direction = np.asarray(ray_direction, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    is_single_ray = ray_origin.ndim == 1
    if is_single_ray:
        ray_origin = ray_origin[:, np.newaxis]
        ray_direction = ray_direction[:, np.newaxis]
    ndotu = np.einsum('i,ij->j', plane_normal, ray_direction)
    w = ray_origin - plane_point[:, np.newaxis]
    dot_product_normal_w = np.einsum('i,ij->j', plane_normal, w)
    t_intersect_values = np.full_like(ndotu, np.nan)
    valid_ndotu_mask = np.abs(ndotu) >= 1e-9
    if np.any(valid_ndotu_mask):
      t_intersect_values[valid_ndotu_mask] = -dot_product_normal_w[valid_ndotu_mask] / ndotu[valid_ndotu_mask]
    valid_t_mask = t_intersect_values >= -1e-7
    intersection_points_3d = np.full_like(ray_origin, np.nan)
    if np.any(valid_t_mask):
        valid_indices = np.where(valid_t_mask)[0]
        intersection_points_3d[:, valid_indices] = ray_origin[:, valid_indices] + \
            ray_direction[:, valid_indices] * t_intersect_values[valid_indices][np.newaxis, :]
    if is_single_ray:
        return (intersection_points_3d[:,0] if valid_t_mask[0] else None,
                t_intersect_values[0] if valid_t_mask[0] else None)
    else:
        return intersection_points_3d, t_intersect_values

def _check_intersection_disk_analytical(ray_origin_tf: np.ndarray, ray_direction_tf: np.ndarray,
                                       target_params: dict) -> np.ndarray | None:
    intersection_points_3d, t_values = _intersect_ray_plane_analytical(ray_origin_tf, ray_direction_tf)
    if intersection_points_3d is None : return None
    is_single_ray = intersection_points_3d.ndim == 1
    if is_single_ray:
        if np.any(np.isnan(intersection_points_3d)): return None
        px, py, _ = intersection_points_3d
        radius = target_params['diameter'] / 2.0
        return intersection_points_3d if np.hypot(px, py) <= radius + config.SIM_INTERSECTION_TOLERANCE else None
    else:
        px = intersection_points_3d[0,:]; py = intersection_points_3d[1,:]
        radius = target_params['diameter'] / 2.0
        valid_mask = ~np.isnan(px) & (np.hypot(px, py) <= radius + config.SIM_INTERSECTION_TOLERANCE)
        output_points = np.full_like(intersection_points_3d, np.nan)
        output_points[:, valid_mask] = intersection_points_3d[:, valid_mask]
        return output_points

def _check_intersection_dome_analytical(ray_origin_tf: np.ndarray, ray_direction_tf: np.ndarray,
                                      target_params: dict) -> np.ndarray | None:
    is_single_ray = ray_origin_tf.ndim == 1
    if not is_single_ray:
        num_rays = ray_origin_tf.shape[1]
        results = np.full((3, num_rays), np.nan)
        for i in range(num_rays):
            if np.all(np.isnan(ray_origin_tf[:, i])) or np.all(np.isnan(ray_direction_tf[:, i])): continue
            res_single = _check_intersection_dome_analytical(ray_origin_tf[:,i], ray_direction_tf[:,i], target_params)
            if res_single is not None: results[:,i] = res_single
        return results
    R_dome = target_params['dome_radius']; base_diameter = target_params['diameter']
    base_radius = base_diameter / 2.0
    if R_dome <= 1e-6:
        return _check_intersection_disk_analytical(ray_origin_tf, ray_direction_tf, {'diameter': base_diameter}) if base_diameter > 1e-6 else None
    O = np.asarray(ray_origin_tf, dtype=float); D = np.asarray(ray_direction_tf, dtype=float)
    C = np.array([0, 0, -R_dome])
    a = np.dot(D, D); OC = O - C; b = 2 * np.dot(D, OC); c_sphere = np.dot(OC, OC) - R_dome**2
    discriminant = b**2 - 4*a*c_sphere
    if discriminant < -1e-9: return None
    if discriminant < 0: discriminant = 0
    sqrt_discriminant = math.sqrt(discriminant)
    t_values = []
    if abs(a) > 1e-9:
        t1 = (-b - sqrt_discriminant) / (2*a); t2 = (-b + sqrt_discriminant) / (2*a)
        if t1 >= -1e-7: t_values.append(t1)
        if t2 >= -1e-7 and not np.isclose(t1,t2): t_values.append(t2)
    if not t_values: return None
    t_values.sort()
    for t_val in t_values:
        p_intersect = O + t_val * D
        if p_intersect[2] + R_dome < -config.SIM_INTERSECTION_TOLERANCE: continue
        if np.hypot(p_intersect[0], p_intersect[1]) <= base_radius + config.SIM_INTERSECTION_TOLERANCE:
            return p_intersect
    return None

def _check_intersection_linear_analytical(ray_origin_tf: np.ndarray, ray_direction_tf: np.ndarray,
                                         target_params: dict) -> np.ndarray | None:
    intersection_points_3d, t_values = _intersect_ray_plane_analytical(ray_origin_tf, ray_direction_tf)
    if intersection_points_3d is None: return None
    is_single_ray = intersection_points_3d.ndim == 1
    if is_single_ray:
        if np.any(np.isnan(intersection_points_3d)): return None
        px, py, _ = intersection_points_3d
        half_length = target_params['length'] / 2.0; half_width = target_params['width'] / 2.0
        return intersection_points_3d if (
            -half_length - config.SIM_INTERSECTION_TOLERANCE <= px <= half_length + config.SIM_INTERSECTION_TOLERANCE and
            -half_width - config.SIM_INTERSECTION_TOLERANCE <= py <= half_width + config.SIM_INTERSECTION_TOLERANCE
        ) else None
    else:
        px = intersection_points_3d[0,:]; py = intersection_points_3d[1,:]
        half_length = target_params['length'] / 2.0; half_width = target_params['width'] / 2.0
        valid_mask = (~np.isnan(px) &
                      (px >= -half_length - config.SIM_INTERSECTION_TOLERANCE) &
                      (px <= half_length + config.SIM_INTERSECTION_TOLERANCE) &
                      (py >= -half_width - config.SIM_INTERSECTION_TOLERANCE) &
                      (py <= half_width + config.SIM_INTERSECTION_TOLERANCE))
        output_points = np.full_like(intersection_points_3d, np.nan)
        output_points[:, valid_mask] = intersection_points_3d[:, valid_mask]
        return output_points

def _check_intersection_planetary_disk_analytical(ray_origin_tf: np.ndarray, ray_direction_tf: np.ndarray,
                                                 target_params: dict) -> np.ndarray | None:
    intersection_points_3d, t_values = _intersect_ray_plane_analytical(ray_origin_tf, ray_direction_tf)
    if intersection_points_3d is None: return None
    is_single_ray = intersection_points_3d.ndim == 1
    if is_single_ray:
        if np.any(np.isnan(intersection_points_3d)): return None
        px, py, _ = intersection_points_3d
        disk_radius = target_params['planet_diameter'] / 2.0
        return intersection_points_3d if np.hypot(px, py) <= disk_radius + config.SIM_INTERSECTION_TOLERANCE else None
    else:
        px = intersection_points_3d[0,:]; py = intersection_points_3d[1,:]
        disk_radius = target_params['planet_diameter'] / 2.0
        valid_mask = ~np.isnan(px) & (np.hypot(px, py) <= disk_radius + config.SIM_INTERSECTION_TOLERANCE)
        output_points = np.full_like(intersection_points_3d, np.nan)
        output_points[:, valid_mask] = intersection_points_3d[:, valid_mask]
        return output_points

def _calculate_transforms_disk_dome(t: float, p: dict):
    omega = 2 * math.pi * p['rpm'] / 60.0; target_angle = omega * t
    target_rot_inv = rotation_matrix([0, 0, 1], -target_angle)
    src_base = np.array([p['src_x'], p['src_y'], p['src_z']])
    rot_x_rad = math.radians(p['rot_x']); rot_y_rad = math.radians(p['rot_y'])
    mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad); mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
    source_rot = mat_rot_x @ mat_rot_y
    return src_base, source_rot, target_rot_inv

def _calculate_transforms_linear(t: float, p: dict):
    physical_length = p['length']
    source_x_world = p['src_x']
    overrun_value = abs(p.get('src_x', 0.0)) 
    src_base = np.array([source_x_world, p['src_y'], p['src_z']])
    rot_x_rad = math.radians(p['rot_x']); rot_y_rad = math.radians(p['rot_y'])
    mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad); mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
    source_rot = mat_rot_x @ mat_rot_y
    speed = p['speed']
    total_dist_travelled_by_center = speed * t
    min_center_substrate_x = source_x_world - (physical_length / 2.0) - overrun_value
    max_center_substrate_x = source_x_world + (physical_length / 2.0) + overrun_value
    travel_length_one_way = max_center_substrate_x - min_center_substrate_x
    cycle_path_length = 2 * travel_length_one_way
    if cycle_path_length <= 1e-9: target_offset_x = source_x_world
    else:
        current_pos_in_cycle = total_dist_travelled_by_center % cycle_path_length
        if current_pos_in_cycle < travel_length_one_way:
            target_offset_x = min_center_substrate_x + current_pos_in_cycle
        else:
            target_offset_x = max_center_substrate_x - (current_pos_in_cycle - travel_length_one_way)
    target_offset = np.array([target_offset_x, 0.0, 0.0])
    return src_base, source_rot, target_offset

def _calculate_transforms_planetary(t: float, p: dict):
    orbital_radius = p['orbit_diameter'] / 2.0
    omega_orb = 2 * math.pi * p['rpm_orbit'] / 60.0; omega_self = 2 * math.pi * p['rpm_disk'] / 60.0
    angle_orbit = omega_orb * t; angle_self = omega_self * t
    target_center_x = orbital_radius * math.cos(angle_orbit); target_center_y = orbital_radius * math.sin(angle_orbit)
    target_center_at_t = np.array([target_center_x, target_center_y, 0.0])
    inv_target_self_rot_matrix = rotation_matrix([0, 0, 1], -angle_self)
    src_base = np.array([p['src_x'], p['src_y'], p['src_z']])
    rot_x_rad = math.radians(p['rot_x']); rot_y_rad = math.radians(p['rot_y'])
    mat_rot_y = rotation_matrix([0, 1, 0], rot_y_rad); mat_rot_x = rotation_matrix([1, 0, 0], rot_x_rad)
    source_rot = mat_rot_x @ mat_rot_y
    return src_base, source_rot, (target_center_at_t, inv_target_self_rot_matrix)

def _simulation_mp_worker(args_tuple):
    worker_id, start_particle_idx, num_particles_in_chunk, params, \
    calculate_transforms_for_analytical, check_intersection_analytical, \
    progress_q, cancel_event_is_set_func = args_tuple

    # grid_size теперь означает количество ГРАНИЦ ячеек
    num_edges = config.SIM_GRID_SIZE 
    num_cells = num_edges - 1

    target_type = params.get('target_type')

    if target_type == config.TARGET_LINEAR:
        grid_x_radius = params.get('length', 1.0) / 2.0
        grid_y_radius = params.get('width', 1.0) / 2.0
    elif target_type == config.TARGET_PLANETARY:
        grid_x_radius = params.get('planet_diameter', 1.0) / 2.0
        grid_y_radius = grid_x_radius
    else: # Disk, Dome
        grid_x_radius = params.get('diameter', 1.0) / 2.0
        grid_y_radius = grid_x_radius
    
    if grid_x_radius <=0 or grid_y_radius <=0:
        grid_x_radius = max(grid_x_radius, 1.0)
        grid_y_radius = max(grid_y_radius, 1.0)

    # Координаты ГРАНИЦ ячеек
    x_coords_edges = np.linspace(-grid_x_radius, grid_x_radius, num_edges)
    y_coords_edges = np.linspace(-grid_y_radius, grid_y_radius, num_edges)
    
    # Карта покрытия теперь имеет размер (количество ячеек Y, количество ячеек X)
    local_coverage_map = np.zeros((num_cells, num_cells), dtype=np.int32)

    # Границы всей сетки
    x_grid_min_edge = x_coords_edges[0]
    x_grid_max_edge = x_coords_edges[-1]
    y_grid_min_edge = y_coords_edges[0]
    y_grid_max_edge = y_coords_edges[-1]
    
    particles_processed_in_chunk_total = 0
    if progress_q:
        progress_q.put((worker_id, 0, num_particles_in_chunk))
    
    outer_vectorized_batch_size = 1024 
    mini_batch_size_for_transforms = params.get('mini_batch_size', 64)
    if mini_batch_size_for_transforms <= 0:
        mini_batch_size_for_transforms = 64 
    
    report_chunk_interval_counter = 0
    report_after_N_particles_processed = max(1, num_particles_in_chunk // 20)

    for i_outer_batch_start in range(0, num_particles_in_chunk, outer_vectorized_batch_size):
        if cancel_event_is_set_func and cancel_event_is_set_func(): break
        
        current_outer_batch_size = min(outer_vectorized_batch_size, num_particles_in_chunk - i_outer_batch_start)
        if current_outer_batch_size <= 0: break

        global_particle_indices_outer = np.arange(
            start_particle_idx + i_outer_batch_start, 
            start_particle_idx + i_outer_batch_start + current_outer_batch_size
        )
        total_particles_for_time = params.get('total_particles_for_time_calc', params['particles'])

        local_src_offsets_array_outer = sample_source_position_vectorized(params['src_type'], params, current_outer_batch_size)
        local_dir_vectors_array_outer = sample_emission_vector_vectorized(params['dist_type'], params['max_theta'], params, current_outer_batch_size)

        for i_mini_batch_start_in_outer in range(0, current_outer_batch_size, mini_batch_size_for_transforms):
            if cancel_event_is_set_func and cancel_event_is_set_func(): break

            current_mini_batch_size = min(mini_batch_size_for_transforms, current_outer_batch_size - i_mini_batch_start_in_outer)
            if current_mini_batch_size <= 0: break

            mini_batch_slice = slice(i_mini_batch_start_in_outer, i_mini_batch_start_in_outer + current_mini_batch_size)
            first_global_idx_in_mini_batch = global_particle_indices_outer[i_mini_batch_start_in_outer]
            t_first_in_mini_batch = params['time'] * first_global_idx_in_mini_batch / total_particles_for_time if total_particles_for_time > 1 else 0
            
            src_pos_base_global, source_rotation_matrix, target_transform_info = \
                calculate_transforms_for_analytical(t_first_in_mini_batch, params)

            current_local_src_offsets = local_src_offsets_array_outer[:, mini_batch_slice]
            current_local_dir_vectors = local_dir_vectors_array_outer[:, mini_batch_slice]

            global_ray_origins_array = src_pos_base_global[:, np.newaxis] + source_rotation_matrix @ current_local_src_offsets
            global_ray_directions_array = source_rotation_matrix @ current_local_dir_vectors
            
            norms = np.linalg.norm(global_ray_directions_array, axis=0)
            valid_norms_mask = norms > 1e-9
            global_ray_directions_array[:, valid_norms_mask] /= norms[valid_norms_mask][np.newaxis, :]

            if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
                target_rotation_matrix_inv = target_transform_info
                ray_origins_tf_array = target_rotation_matrix_inv @ global_ray_origins_array
                ray_directions_tf_array = target_rotation_matrix_inv @ global_ray_directions_array
            elif target_type == config.TARGET_LINEAR:
                target_offset_at_t = target_transform_info
                ray_origins_tf_array = global_ray_origins_array - target_offset_at_t[:, np.newaxis]
                ray_directions_tf_array = global_ray_directions_array
            elif target_type == config.TARGET_PLANETARY:
                target_center_at_t, inv_target_self_rot_matrix = target_transform_info
                ray_origins_rel_center_array = global_ray_origins_array - target_center_at_t[:, np.newaxis]
                ray_origins_tf_array = inv_target_self_rot_matrix @ ray_origins_rel_center_array
                ray_directions_tf_array = inv_target_self_rot_matrix @ global_ray_directions_array
            else:
                ray_origins_tf_array = global_ray_origins_array
                ray_directions_tf_array = global_ray_directions_array

            intersection_points_on_target_3d_array = check_intersection_analytical(
                ray_origins_tf_array, ray_directions_tf_array, params
            )

            if intersection_points_on_target_3d_array is not None:
                valid_hits_mask_initial = ~np.isnan(intersection_points_on_target_3d_array[0,:])
                if np.any(valid_hits_mask_initial):
                    px_hits_all = intersection_points_on_target_3d_array[0, valid_hits_mask_initial]
                    py_hits_all = intersection_points_on_target_3d_array[1, valid_hits_mask_initial]
                    
                    # Проверка попадания в ОБЩИЕ ГРАНИЦЫ СЕТКИ
                    # Частицы на самой правой/верхней границе исключаются (стандарт для гистограмм)
                    grid_bounds_mask = (
                        (px_hits_all >= x_grid_min_edge) & (px_hits_all < x_grid_max_edge) &
                        (py_hits_all >= y_grid_min_edge) & (py_hits_all < y_grid_max_edge)
                    )
                    
                    px_hits_in_grid = px_hits_all[grid_bounds_mask]
                    py_hits_in_grid = py_hits_all[grid_bounds_mask]

                    if px_hits_in_grid.size > 0:
                        # searchsorted с границами ячеек возвращает индекс ячейки
                        ix_hits = np.searchsorted(x_coords_edges, px_hits_in_grid, side='right') - 1
                        iy_hits = np.searchsorted(y_coords_edges, py_hits_in_grid, side='right') - 1
                        
                        # Клиппинг по индексам ячеек (0 до num_cells - 1)
                        ix_hits = np.clip(ix_hits, 0, num_cells - 1)
                        iy_hits = np.clip(iy_hits, 0, num_cells - 1)
                        
                        np.add.at(local_coverage_map, (iy_hits, ix_hits), 1)
            
            particles_processed_in_chunk_total += current_mini_batch_size
            report_chunk_interval_counter += current_mini_batch_size

            if progress_q and (report_chunk_interval_counter >= report_after_N_particles_processed or \
                               particles_processed_in_chunk_total == num_particles_in_chunk) :
                 progress_q.put((worker_id, particles_processed_in_chunk_total, num_particles_in_chunk))
                 report_chunk_interval_counter = 0

        if cancel_event_is_set_func and cancel_event_is_set_func(): break

    if progress_q:
        progress_q.put((worker_id, particles_processed_in_chunk_total, num_particles_in_chunk))
    return local_coverage_map


def _run_simulation_multiprocessed(params: dict,
                                   calculate_transforms_func,
                                   check_intersection_func,
                                   progress_q=None,
                                   cancel_event=None):
    n_particles_total = params['particles']
    if n_particles_total == 0:
        return _get_empty_map_data(params, progress_q) # Передаем progress_q

    num_cores = os.cpu_count()
    if n_particles_total <= 20000: num_workers = max(1, min(num_cores if num_cores else 1, 2))
    elif n_particles_total <= 100000: num_workers = max(1, (num_cores if num_cores else 1) // 2 if (num_cores if num_cores else 1) > 2 else (num_cores if num_cores else 1))
    else: num_workers = max(1, (num_cores if num_cores else 1) - 1 if (num_cores if num_cores else 1) > 1 else 1)
    min_sensible_chunk_per_worker = 5000
    if num_workers > 0 and n_particles_total / num_workers < min_sensible_chunk_per_worker and num_workers > 1:
        num_workers = max(1, math.ceil(n_particles_total / min_sensible_chunk_per_worker))
    num_workers = min(num_workers, n_particles_total if n_particles_total > 0 else 1)
    num_workers = max(1, num_workers)
    chunk_size = math.ceil(n_particles_total / num_workers) if num_workers > 0 else n_particles_total
    if chunk_size == 0 and n_particles_total > 0 : chunk_size = n_particles_total
    print(f"DEBUG: Всего частиц: {n_particles_total}, Ядер CPU: {num_cores}, Выбрано воркеров: {num_workers}, Размер чанка: {chunk_size}")

    params_with_total_particles = params.copy()
    params_with_total_particles['total_particles_for_time_calc'] = n_particles_total
    tasks_args = []
    current_start_idx = 0
    for i in range(num_workers):
        num_in_this_chunk = min(chunk_size, n_particles_total - current_start_idx)
        if num_in_this_chunk <= 0: break
        tasks_args.append((
            i, current_start_idx, num_in_this_chunk, params_with_total_particles,
            calculate_transforms_func, check_intersection_func,
            progress_q, cancel_event.is_set if cancel_event else lambda: False
        ))
        current_start_idx += num_in_this_chunk
    
    list_of_local_coverage_maps = []
    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_workers)
        async_results = pool.map_async(_simulation_mp_worker, tasks_args)
        while not async_results.ready():
            if cancel_event and cancel_event.is_set():
                print("Отмена симуляции (замечено в _run_simulation_multiprocessed)")
                pool.terminate(); pool.join()
                return _get_empty_map_data(params)
            time.sleep(0.05)
        pool.close(); pool.join()
        if async_results.successful(): list_of_local_coverage_maps = async_results.get()
        else:
            try: async_results.get()
            except Exception as e_worker: print(f"Ошибка из воркера: {e_worker}"); raise e_worker
            raise RuntimeError("Ошибка выполнения в одном из дочерних процессов симуляции (неизвестная).")
    except (KeyboardInterrupt, SystemExit):
        print("Симуляция прервана KeyboardInterrupt/SystemExit")
        if pool is not None: pool.terminate(); pool.join()
        return _get_empty_map_data(params)
    except Exception as e_pool:
        print(f"Общая ошибка при работе с пулом: {e_pool}")
        if pool is not None: pool.terminate(); pool.join()
        raise e_pool

    if not list_of_local_coverage_maps or not any(m is not None for m in list_of_local_coverage_maps):
        print("Не получено карт покрытия от воркеров или все они None.")
        return _get_empty_map_data(params)

    # Суммируем карты покрытия, они уже правильного размера (num_cells, num_cells)
    final_coverage_map = np.sum(np.array([m for m in list_of_local_coverage_maps if m is not None]), axis=0).astype(np.int32)
    
    # --- Возвращаем ГРАНИЦЫ ячеек и RADIUS_GRID на основе ЦЕНТРОВ ячеек ---
    num_edges_final = config.SIM_GRID_SIZE
    num_cells_final = num_edges_final - 1
    target_type_final = params.get('target_type')

    if target_type_final == config.TARGET_LINEAR:
        grid_x_rad_final = params.get('length', 1.0) / 2.0
        grid_y_rad_final = params.get('width', 1.0) / 2.0
    elif target_type_final == config.TARGET_PLANETARY:
        grid_x_rad_final = params.get('planet_diameter', 1.0) / 2.0
        grid_y_rad_final = grid_x_rad_final
    else: # Disk, Dome
        grid_x_rad_final = params.get('diameter', 1.0) / 2.0
        grid_y_rad_final = grid_x_rad_final
    
    if grid_x_rad_final <=0 or grid_y_rad_final <=0:
        grid_x_rad_final = max(grid_x_rad_final, 1.0); grid_y_rad_final = max(grid_y_rad_final, 1.0)

    x_coords_edges_final = np.linspace(-grid_x_rad_final, grid_x_rad_final, num_edges_final)
    y_coords_edges_final = np.linspace(-grid_y_rad_final, grid_y_rad_final, num_edges_final)
    
    # Координаты центров ячеек
    x_centers_final = (x_coords_edges_final[:-1] + x_coords_edges_final[1:]) / 2.0
    y_centers_final = (y_coords_edges_final[:-1] + y_coords_edges_final[1:]) / 2.0
    
    xx_centers, yy_centers = np.meshgrid(x_centers_final, y_centers_final) # Meshgrid для центров
    radius_grid_at_centers = np.hypot(xx_centers, yy_centers) # (num_cells_y, num_cells_x)
    
    # Проверка формы final_coverage_map
    if final_coverage_map.shape[0] != num_cells_final or final_coverage_map.shape[1] != num_cells_final:
        print(f"Предупреждение: форма final_coverage_map ({final_coverage_map.shape}) не совпадает с ожидаемой ({num_cells_final}, {num_cells_final}).")
        # Попытка изменить форму, если общее количество элементов совпадает
        if final_coverage_map.size == num_cells_final * num_cells_final:
            try:
                final_coverage_map = final_coverage_map.reshape((num_cells_final, num_cells_final))
                print("Форма карты покрытия была скорректирована.")
            except ValueError:
                print("Не удалось скорректировать форму карты покрытия. Возвращаем пустую карту.")
                return _get_empty_map_data(params)
        else:
            print("Размер карты покрытия не соответствует ожидаемому. Возвращаем пустую карту.")
            return _get_empty_map_data(params)
            
    return final_coverage_map, x_coords_edges_final, y_coords_edges_final, radius_grid_at_centers

def _get_empty_map_data(params: dict, progress_q=None) -> tuple: # Добавлен progress_q
    num_edges = config.SIM_GRID_SIZE
    num_cells = num_edges - 1
    target_type = params.get('target_type')

    if target_type == config.TARGET_LINEAR:
        grid_x_radius = params.get('length', 1.0) / 2.0; grid_y_radius = params.get('width', 1.0) / 2.0
    elif target_type == config.TARGET_PLANETARY:
        grid_x_radius = params.get('planet_diameter', 1.0) / 2.0; grid_y_radius = grid_x_radius
    else: # Disk, Dome
        grid_x_radius = params.get('diameter', 1.0) / 2.0; grid_y_radius = grid_x_radius
    
    if grid_x_radius <=0 or grid_y_radius <=0:
        grid_x_radius = max(grid_x_radius, 1.0); grid_y_radius = max(grid_y_radius, 1.0)
    
    x_coords_edges = np.linspace(-grid_x_radius, grid_x_radius, num_edges)
    y_coords_edges = np.linspace(-grid_y_radius, grid_y_radius, num_edges)
    
    coverage_map = np.zeros((num_cells, num_cells), dtype=np.int32) # Карта размера ячеек
    
    x_centers = (x_coords_edges[:-1] + x_coords_edges[1:]) / 2.0
    y_centers = (y_coords_edges[:-1] + y_coords_edges[1:]) / 2.0
    xx_centers, yy_centers = np.meshgrid(x_centers, y_centers)
    radius_grid_at_centers = np.hypot(xx_centers, yy_centers)
    
    if progress_q and hasattr(progress_q, 'put'): progress_q.put(100) # Если 0 частиц, считаем завершенным

    return coverage_map, x_coords_edges, y_coords_edges, radius_grid_at_centers

def simulate_coating_disk_dome_mp(params: dict, progress_q=None, cancel_event=None):
    target_type = params['target_type']
    calculate_transforms_func = _calculate_transforms_disk_dome
    check_func = _check_intersection_dome_analytical if target_type == config.TARGET_DOME else _check_intersection_disk_analytical
    params.setdefault('rpm', config.DEFAULT_PROCESSING_PARAMS['rpm'])
    params.setdefault('mini_batch_size', config.DEFAULT_PROCESSING_PARAMS['mini_batch_size'])
    return _run_simulation_multiprocessed(params, calculate_transforms_func, check_func, progress_q, cancel_event)

def simulate_linear_movement_mp(params: dict, progress_q=None, cancel_event=None):
    calculate_transforms_func = _calculate_transforms_linear
    params.setdefault('speed', config.DEFAULT_PROCESSING_PARAMS['speed'])
    params.setdefault('src_x', 0.0) 
    params.setdefault('mini_batch_size', config.DEFAULT_PROCESSING_PARAMS['mini_batch_size'])
    return _run_simulation_multiprocessed(params, calculate_transforms_func, _check_intersection_linear_analytical, progress_q, cancel_event)

def simulate_planetary_mp(params: dict, progress_q=None, cancel_event=None):
    calculate_transforms_func = _calculate_transforms_planetary
    params.setdefault('rpm_disk', config.DEFAULT_PROCESSING_PARAMS['rpm_disk'])
    params.setdefault('rpm_orbit', config.DEFAULT_PROCESSING_PARAMS['rpm_orbit'])
    params.setdefault('mini_batch_size', config.DEFAULT_PROCESSING_PARAMS['mini_batch_size'])
    return _run_simulation_multiprocessed(params, calculate_transforms_func, _check_intersection_planetary_disk_analytical, progress_q, cancel_event)

