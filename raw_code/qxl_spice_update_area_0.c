void qxl_spice_update_area(PCIQXLDevice *qxl, uint32_t surface_id,

                           struct QXLRect *area, struct QXLRect *dirty_rects,

                           uint32_t num_dirty_rects,

                           uint32_t clear_dirty_region,

                           qxl_async_io async)

{

    if (async == QXL_SYNC) {

        qxl->ssd.worker->update_area(qxl->ssd.worker, surface_id, area,

                        dirty_rects, num_dirty_rects, clear_dirty_region);

    } else {

#if SPICE_INTERFACE_QXL_MINOR >= 1

        spice_qxl_update_area_async(&qxl->ssd.qxl, surface_id, area,

                                    clear_dirty_region, 0);

#else

        abort();

#endif

    }

}
