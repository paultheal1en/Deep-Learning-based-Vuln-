static void qxl_render_update_area_unlocked(PCIQXLDevice *qxl)

{

    VGACommonState *vga = &qxl->vga;

    int i;



    if (qxl->guest_primary.resized) {

        qxl->guest_primary.resized = 0;

        qxl->guest_primary.data = memory_region_get_ram_ptr(&qxl->vga.vram);

        qxl_set_rect_to_surface(qxl, &qxl->dirty[0]);

        qxl->num_dirty_rects = 1;

        trace_qxl_render_guest_primary_resized(

               qxl->guest_primary.surface.width,

               qxl->guest_primary.surface.height,

               qxl->guest_primary.qxl_stride,

               qxl->guest_primary.bytes_pp,

               qxl->guest_primary.bits_pp);

        if (qxl->guest_primary.qxl_stride > 0) {

            qemu_free_displaysurface(vga->ds);

            qemu_create_displaysurface_from(qxl->guest_primary.surface.width,

                                            qxl->guest_primary.surface.height,

                                            qxl->guest_primary.bits_pp,

                                            qxl->guest_primary.abs_stride,

                                            qxl->guest_primary.data);

        } else {

            qemu_resize_displaysurface(vga->ds,

                    qxl->guest_primary.surface.width,

                    qxl->guest_primary.surface.height);

        }

        dpy_gfx_resize(vga->ds);

    }

    for (i = 0; i < qxl->num_dirty_rects; i++) {

        if (qemu_spice_rect_is_empty(qxl->dirty+i)) {

            break;

        }

        qxl_blit(qxl, qxl->dirty+i);

        dpy_gfx_update(vga->ds,

                       qxl->dirty[i].left, qxl->dirty[i].top,

                       qxl->dirty[i].right - qxl->dirty[i].left,

                       qxl->dirty[i].bottom - qxl->dirty[i].top);

    }

    qxl->num_dirty_rects = 0;

}
