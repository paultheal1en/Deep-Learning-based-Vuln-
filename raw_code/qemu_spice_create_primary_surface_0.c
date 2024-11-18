void qemu_spice_create_primary_surface(SimpleSpiceDisplay *ssd, uint32_t id,

                                       QXLDevSurfaceCreate *surface,

                                       qxl_async_io async)

{

    if (async != QXL_SYNC) {

#if SPICE_INTERFACE_QXL_MINOR >= 1

        spice_qxl_create_primary_surface_async(&ssd->qxl, id, surface, 0);

#else

        abort();

#endif

    } else {

        ssd->worker->create_primary_surface(ssd->worker, id, surface);

    }

}
