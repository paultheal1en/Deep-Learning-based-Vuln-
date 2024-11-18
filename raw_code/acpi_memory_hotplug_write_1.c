static void acpi_memory_hotplug_write(void *opaque, hwaddr addr, uint64_t data,

                                      unsigned int size)

{

    MemHotplugState *mem_st = opaque;

    MemStatus *mdev;

    ACPIOSTInfo *info;

    DeviceState *dev = NULL;

    HotplugHandler *hotplug_ctrl = NULL;

    Error *local_err = NULL;



    if (!mem_st->dev_count) {

        return;

    }



    if (addr) {

        if (mem_st->selector >= mem_st->dev_count) {

            trace_mhp_acpi_invalid_slot_selected(mem_st->selector);

            return;

        }

    }



    switch (addr) {

    case 0x0: /* DIMM slot selector */

        mem_st->selector = data;

        trace_mhp_acpi_write_slot(mem_st->selector);

        break;

    case 0x4: /* _OST event  */

        mdev = &mem_st->devs[mem_st->selector];

        if (data == 1) {

            /* TODO: handle device insert OST event */

        } else if (data == 3) {

            /* TODO: handle device remove OST event */

        }

        mdev->ost_event = data;

        trace_mhp_acpi_write_ost_ev(mem_st->selector, mdev->ost_event);

        break;

    case 0x8: /* _OST status */

        mdev = &mem_st->devs[mem_st->selector];

        mdev->ost_status = data;

        trace_mhp_acpi_write_ost_status(mem_st->selector, mdev->ost_status);

        /* TODO: implement memory removal on guest signal */



        info = acpi_memory_device_status(mem_st->selector, mdev);

        qapi_event_send_acpi_device_ost(info, &error_abort);

        qapi_free_ACPIOSTInfo(info);

        break;

    case 0x14: /* set is_* fields  */

        mdev = &mem_st->devs[mem_st->selector];

        if (data & 2) { /* clear insert event */

            mdev->is_inserting  = false;

            trace_mhp_acpi_clear_insert_evt(mem_st->selector);

        } else if (data & 4) {

            mdev->is_removing = false;

            trace_mhp_acpi_clear_remove_evt(mem_st->selector);

        } else if (data & 8) {

            if (!mdev->is_enabled) {

                trace_mhp_acpi_ejecting_invalid_slot(mem_st->selector);

                break;

            }



            dev = DEVICE(mdev->dimm);

            hotplug_ctrl = qdev_get_hotplug_handler(dev);

            /* call pc-dimm unplug cb */

            hotplug_handler_unplug(hotplug_ctrl, dev, &local_err);

            if (local_err) {

                trace_mhp_acpi_pc_dimm_delete_failed(mem_st->selector);

                qapi_event_send_mem_unplug_error(dev->id,

                                                 error_get_pretty(local_err),

                                                 &error_abort);


                break;

            }

            trace_mhp_acpi_pc_dimm_deleted(mem_st->selector);

        }

        break;

    default:

        break;

    }



}