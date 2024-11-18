static void ioport_write(void *opaque, target_phys_addr_t addr,

                         uint64_t val, unsigned size)

{

    PCIQXLDevice *d = opaque;

    uint32_t io_port = addr;

    qxl_async_io async = QXL_SYNC;

#if SPICE_INTERFACE_QXL_MINOR >= 1

    uint32_t orig_io_port = io_port;

#endif



    switch (io_port) {

    case QXL_IO_RESET:

    case QXL_IO_SET_MODE:

    case QXL_IO_MEMSLOT_ADD:

    case QXL_IO_MEMSLOT_DEL:

    case QXL_IO_CREATE_PRIMARY:

    case QXL_IO_UPDATE_IRQ:

    case QXL_IO_LOG:

#if SPICE_INTERFACE_QXL_MINOR >= 1

    case QXL_IO_MEMSLOT_ADD_ASYNC:

    case QXL_IO_CREATE_PRIMARY_ASYNC:

#endif

        break;

    default:

        if (d->mode != QXL_MODE_VGA) {

            break;

        }

        dprint(d, 1, "%s: unexpected port 0x%x (%s) in vga mode\n",

            __func__, io_port, io_port_to_string(io_port));

#if SPICE_INTERFACE_QXL_MINOR >= 1

        /* be nice to buggy guest drivers */

        if (io_port >= QXL_IO_UPDATE_AREA_ASYNC &&

            io_port <= QXL_IO_DESTROY_ALL_SURFACES_ASYNC) {

            qxl_send_events(d, QXL_INTERRUPT_IO_CMD);

        }

#endif

        return;

    }



#if SPICE_INTERFACE_QXL_MINOR >= 1

    /* we change the io_port to avoid ifdeffery in the main switch */

    orig_io_port = io_port;

    switch (io_port) {

    case QXL_IO_UPDATE_AREA_ASYNC:

        io_port = QXL_IO_UPDATE_AREA;

        goto async_common;

    case QXL_IO_MEMSLOT_ADD_ASYNC:

        io_port = QXL_IO_MEMSLOT_ADD;

        goto async_common;

    case QXL_IO_CREATE_PRIMARY_ASYNC:

        io_port = QXL_IO_CREATE_PRIMARY;

        goto async_common;

    case QXL_IO_DESTROY_PRIMARY_ASYNC:

        io_port = QXL_IO_DESTROY_PRIMARY;

        goto async_common;

    case QXL_IO_DESTROY_SURFACE_ASYNC:

        io_port = QXL_IO_DESTROY_SURFACE_WAIT;

        goto async_common;

    case QXL_IO_DESTROY_ALL_SURFACES_ASYNC:

        io_port = QXL_IO_DESTROY_ALL_SURFACES;

        goto async_common;

    case QXL_IO_FLUSH_SURFACES_ASYNC:

async_common:

        async = QXL_ASYNC;

        qemu_mutex_lock(&d->async_lock);

        if (d->current_async != QXL_UNDEFINED_IO) {

            qxl_guest_bug(d, "%d async started before last (%d) complete",

                io_port, d->current_async);

            qemu_mutex_unlock(&d->async_lock);

            return;

        }

        d->current_async = orig_io_port;

        qemu_mutex_unlock(&d->async_lock);

        dprint(d, 2, "start async %d (%"PRId64")\n", io_port, val);

        break;

    default:

        break;

    }

#endif



    switch (io_port) {

    case QXL_IO_UPDATE_AREA:

    {

        QXLRect update = d->ram->update_area;

        qxl_spice_update_area(d, d->ram->update_surface,

                              &update, NULL, 0, 0, async);

        break;

    }

    case QXL_IO_NOTIFY_CMD:

        qemu_spice_wakeup(&d->ssd);

        break;

    case QXL_IO_NOTIFY_CURSOR:

        qemu_spice_wakeup(&d->ssd);

        break;

    case QXL_IO_UPDATE_IRQ:

        qxl_update_irq(d);

        break;

    case QXL_IO_NOTIFY_OOM:

        if (!SPICE_RING_IS_EMPTY(&d->ram->release_ring)) {

            break;

        }

        d->oom_running = 1;

        qxl_spice_oom(d);

        d->oom_running = 0;

        break;

    case QXL_IO_SET_MODE:

        dprint(d, 1, "QXL_SET_MODE %d\n", (int)val);

        qxl_set_mode(d, val, 0);

        break;

    case QXL_IO_LOG:

        if (d->guestdebug) {

            fprintf(stderr, "qxl/guest-%d: %" PRId64 ": %s", d->id,

                    qemu_get_clock_ns(vm_clock), d->ram->log_buf);

        }

        break;

    case QXL_IO_RESET:

        dprint(d, 1, "QXL_IO_RESET\n");

        qxl_hard_reset(d, 0);

        break;

    case QXL_IO_MEMSLOT_ADD:

        if (val >= NUM_MEMSLOTS) {

            qxl_guest_bug(d, "QXL_IO_MEMSLOT_ADD: val out of range");

            break;

        }

        if (d->guest_slots[val].active) {

            qxl_guest_bug(d, "QXL_IO_MEMSLOT_ADD: memory slot already active");

            break;

        }

        d->guest_slots[val].slot = d->ram->mem_slot;

        qxl_add_memslot(d, val, 0, async);

        break;

    case QXL_IO_MEMSLOT_DEL:

        if (val >= NUM_MEMSLOTS) {

            qxl_guest_bug(d, "QXL_IO_MEMSLOT_DEL: val out of range");

            break;

        }

        qxl_del_memslot(d, val);

        break;

    case QXL_IO_CREATE_PRIMARY:

        if (val != 0) {

            qxl_guest_bug(d, "QXL_IO_CREATE_PRIMARY (async=%d): val != 0",

                          async);

            goto cancel_async;

        }

        dprint(d, 1, "QXL_IO_CREATE_PRIMARY async=%d\n", async);

        d->guest_primary.surface = d->ram->create_surface;

        qxl_create_guest_primary(d, 0, async);

        break;

    case QXL_IO_DESTROY_PRIMARY:

        if (val != 0) {

            qxl_guest_bug(d, "QXL_IO_DESTROY_PRIMARY (async=%d): val != 0",

                          async);

            goto cancel_async;

        }

        dprint(d, 1, "QXL_IO_DESTROY_PRIMARY (async=%d) (%s)\n", async,

               qxl_mode_to_string(d->mode));

        if (!qxl_destroy_primary(d, async)) {

            dprint(d, 1, "QXL_IO_DESTROY_PRIMARY_ASYNC in %s, ignored\n",

                    qxl_mode_to_string(d->mode));

            goto cancel_async;

        }

        break;

    case QXL_IO_DESTROY_SURFACE_WAIT:

        if (val >= NUM_SURFACES) {

            qxl_guest_bug(d, "QXL_IO_DESTROY_SURFACE (async=%d):"

                             "%d >= NUM_SURFACES", async, val);

            goto cancel_async;

        }

        qxl_spice_destroy_surface_wait(d, val, async);

        break;

#if SPICE_INTERFACE_QXL_MINOR >= 1

    case QXL_IO_FLUSH_RELEASE: {

        QXLReleaseRing *ring = &d->ram->release_ring;

        if (ring->prod - ring->cons + 1 == ring->num_items) {

            fprintf(stderr,

                "ERROR: no flush, full release ring [p%d,%dc]\n",

                ring->prod, ring->cons);

        }

        qxl_push_free_res(d, 1 /* flush */);

        dprint(d, 1, "QXL_IO_FLUSH_RELEASE exit (%s, s#=%d, res#=%d,%p)\n",

            qxl_mode_to_string(d->mode), d->guest_surfaces.count,

            d->num_free_res, d->last_release);

        break;

    }

    case QXL_IO_FLUSH_SURFACES_ASYNC:

        dprint(d, 1, "QXL_IO_FLUSH_SURFACES_ASYNC"

                     " (%"PRId64") (%s, s#=%d, res#=%d)\n",

               val, qxl_mode_to_string(d->mode), d->guest_surfaces.count,

               d->num_free_res);

        qxl_spice_flush_surfaces_async(d);

        break;

#endif

    case QXL_IO_DESTROY_ALL_SURFACES:

        d->mode = QXL_MODE_UNDEFINED;

        qxl_spice_destroy_surfaces(d, async);

        break;

    default:

        fprintf(stderr, "%s: ioport=0x%x, abort()\n", __FUNCTION__, io_port);

        abort();

    }

    return;

cancel_async:

#if SPICE_INTERFACE_QXL_MINOR >= 1

    if (async) {

        qxl_send_events(d, QXL_INTERRUPT_IO_CMD);

        qemu_mutex_lock(&d->async_lock);

        d->current_async = QXL_UNDEFINED_IO;

        qemu_mutex_unlock(&d->async_lock);

    }

#else

    return;

#endif

}
