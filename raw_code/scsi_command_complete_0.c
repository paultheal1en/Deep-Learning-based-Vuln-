static void scsi_command_complete(void *opaque, int ret)

{

    int status;

    SCSIGenericReq *r = (SCSIGenericReq *)opaque;



    r->req.aiocb = NULL;

    if (r->req.io_canceled) {

        goto done;

    }

    if (r->io_header.driver_status & SG_ERR_DRIVER_SENSE) {

        r->req.sense_len = r->io_header.sb_len_wr;

    }



    if (ret != 0) {

        switch (ret) {

        case -EDOM:

            status = TASK_SET_FULL;

            break;

        case -ENOMEM:

            status = CHECK_CONDITION;

            scsi_req_build_sense(&r->req, SENSE_CODE(TARGET_FAILURE));

            break;

        default:

            status = CHECK_CONDITION;

            scsi_req_build_sense(&r->req, SENSE_CODE(IO_ERROR));

            break;

        }

    } else {

        if (r->io_header.host_status == SG_ERR_DID_NO_CONNECT ||

            r->io_header.host_status == SG_ERR_DID_BUS_BUSY ||

            r->io_header.host_status == SG_ERR_DID_TIME_OUT ||

            (r->io_header.driver_status & SG_ERR_DRIVER_TIMEOUT)) {

            status = BUSY;

            BADF("Driver Timeout\n");

        } else if (r->io_header.host_status) {

            status = CHECK_CONDITION;

            scsi_req_build_sense(&r->req, SENSE_CODE(I_T_NEXUS_LOSS));

        } else if (r->io_header.status) {

            status = r->io_header.status;

        } else if (r->io_header.driver_status & SG_ERR_DRIVER_SENSE) {

            status = CHECK_CONDITION;

        } else {

            status = GOOD;

        }

    }

    DPRINTF("Command complete 0x%p tag=0x%x status=%d\n",

            r, r->req.tag, status);



    scsi_req_complete(&r->req, status);

done:

    if (!r->req.io_canceled) {

        scsi_req_unref(&r->req);

    }

}
