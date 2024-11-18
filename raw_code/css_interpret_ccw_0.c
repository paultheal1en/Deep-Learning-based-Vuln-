static int css_interpret_ccw(SubchDev *sch, hwaddr ccw_addr,

                             bool suspend_allowed)

{

    int ret;

    bool check_len;

    int len;

    CCW1 ccw;



    if (!ccw_addr) {

        return -EIO;

    }

    /* Check doubleword aligned and 31 or 24 (fmt 0) bit addressable. */

    if (ccw_addr & (sch->ccw_fmt_1 ? 0x80000007 : 0xff000007)) {

        return -EINVAL;

    }



    /* Translate everything to format-1 ccws - the information is the same. */

    ccw = copy_ccw_from_guest(ccw_addr, sch->ccw_fmt_1);



    /* Check for invalid command codes. */

    if ((ccw.cmd_code & 0x0f) == 0) {

        return -EINVAL;

    }

    if (((ccw.cmd_code & 0x0f) == CCW_CMD_TIC) &&

        ((ccw.cmd_code & 0xf0) != 0)) {

        return -EINVAL;

    }

    if (!sch->ccw_fmt_1 && (ccw.count == 0) &&

        (ccw.cmd_code != CCW_CMD_TIC)) {

        return -EINVAL;

    }



    /* We don't support MIDA. */

    if (ccw.flags & CCW_FLAG_MIDA) {

        return -EINVAL;

    }



    if (ccw.flags & CCW_FLAG_SUSPEND) {

        return suspend_allowed ? -EINPROGRESS : -EINVAL;

    }



    check_len = !((ccw.flags & CCW_FLAG_SLI) && !(ccw.flags & CCW_FLAG_DC));



    if (!ccw.cda) {

        if (sch->ccw_no_data_cnt == 255) {

            return -EINVAL;

        }

        sch->ccw_no_data_cnt++;

    }



    /* Look at the command. */

    switch (ccw.cmd_code) {

    case CCW_CMD_NOOP:

        /* Nothing to do. */

        ret = 0;

        break;

    case CCW_CMD_BASIC_SENSE:

        if (check_len) {

            if (ccw.count != sizeof(sch->sense_data)) {

                ret = -EINVAL;

                break;

            }

        }

        len = MIN(ccw.count, sizeof(sch->sense_data));

        cpu_physical_memory_write(ccw.cda, sch->sense_data, len);

        sch->curr_status.scsw.count = ccw.count - len;

        memset(sch->sense_data, 0, sizeof(sch->sense_data));

        ret = 0;

        break;

    case CCW_CMD_SENSE_ID:

    {

        SenseId sense_id;



        copy_sense_id_to_guest(&sense_id, &sch->id);

        /* Sense ID information is device specific. */

        if (check_len) {

            if (ccw.count != sizeof(sense_id)) {

                ret = -EINVAL;

                break;

            }

        }

        len = MIN(ccw.count, sizeof(sense_id));

        /*

         * Only indicate 0xff in the first sense byte if we actually

         * have enough place to store at least bytes 0-3.

         */

        if (len >= 4) {

            sense_id.reserved = 0xff;

        } else {

            sense_id.reserved = 0;

        }

        cpu_physical_memory_write(ccw.cda, &sense_id, len);

        sch->curr_status.scsw.count = ccw.count - len;

        ret = 0;

        break;

    }

    case CCW_CMD_TIC:

        if (sch->last_cmd_valid && (sch->last_cmd.cmd_code == CCW_CMD_TIC)) {

            ret = -EINVAL;

            break;

        }

        if (ccw.flags & (CCW_FLAG_CC | CCW_FLAG_DC)) {

            ret = -EINVAL;

            break;

        }

        sch->channel_prog = ccw.cda;

        ret = -EAGAIN;

        break;

    default:

        if (sch->ccw_cb) {

            /* Handle device specific commands. */

            ret = sch->ccw_cb(sch, ccw);

        } else {

            ret = -ENOSYS;

        }

        break;

    }

    sch->last_cmd = ccw;

    sch->last_cmd_valid = true;

    if (ret == 0) {

        if (ccw.flags & CCW_FLAG_CC) {

            sch->channel_prog += 8;

            ret = -EAGAIN;

        }

    }



    return ret;

}
