static void qxl_spice_destroy_surfaces(PCIQXLDevice *qxl, qxl_async_io async)

{

    if (async) {

#if SPICE_INTERFACE_QXL_MINOR < 1

        abort();

#else

        spice_qxl_destroy_surfaces_async(&qxl->ssd.qxl, 0);

#endif

    } else {

        qxl->ssd.worker->destroy_surfaces(qxl->ssd.worker);

        qxl_spice_destroy_surfaces_complete(qxl);

    }

}
