static void mptsas_process_scsi_task_mgmt(MPTSASState *s, MPIMsgSCSITaskMgmt *req)

{

    MPIMsgSCSITaskMgmtReply reply;

    MPIMsgSCSITaskMgmtReply *reply_async;

    int status, count;

    SCSIDevice *sdev;

    SCSIRequest *r, *next;

    BusChild *kid;



    mptsas_fix_scsi_task_mgmt_endianness(req);



    QEMU_BUILD_BUG_ON(MPTSAS_MAX_REQUEST_SIZE < sizeof(*req));

    QEMU_BUILD_BUG_ON(sizeof(s->doorbell_msg) < sizeof(*req));

    QEMU_BUILD_BUG_ON(sizeof(s->doorbell_reply) < sizeof(reply));



    memset(&reply, 0, sizeof(reply));

    reply.TargetID   = req->TargetID;

    reply.Bus        = req->Bus;

    reply.MsgLength  = sizeof(reply) / 4;

    reply.Function   = req->Function;

    reply.TaskType   = req->TaskType;

    reply.MsgContext = req->MsgContext;



    switch (req->TaskType) {

    case MPI_SCSITASKMGMT_TASKTYPE_ABORT_TASK:

    case MPI_SCSITASKMGMT_TASKTYPE_QUERY_TASK:

        status = mptsas_scsi_device_find(s, req->Bus, req->TargetID,

                                         req->LUN, &sdev);

        if (status) {

            reply.IOCStatus = status;

            goto out;

        }

        if (sdev->lun != req->LUN[1]) {

            reply.ResponseCode = MPI_SCSITASKMGMT_RSP_TM_INVALID_LUN;

            goto out;

        }



        QTAILQ_FOREACH_SAFE(r, &sdev->requests, next, next) {

            MPTSASRequest *cmd_req = r->hba_private;

            if (cmd_req && cmd_req->scsi_io.MsgContext == req->TaskMsgContext) {

                break;

            }

        }

        if (r) {

            /*

             * Assert that the request has not been completed yet, we

             * check for it in the loop above.

             */

            assert(r->hba_private);

            if (req->TaskType == MPI_SCSITASKMGMT_TASKTYPE_QUERY_TASK) {

                /* "If the specified command is present in the task set, then

                 * return a service response set to FUNCTION SUCCEEDED".

                 */

                reply.ResponseCode = MPI_SCSITASKMGMT_RSP_TM_SUCCEEDED;

            } else {

                MPTSASCancelNotifier *notifier;



                reply_async = g_memdup(&reply, sizeof(MPIMsgSCSITaskMgmtReply));

                reply_async->IOCLogInfo = INT_MAX;



                count = 1;

                notifier = g_new(MPTSASCancelNotifier, 1);

                notifier->s = s;

                notifier->reply = reply_async;

                notifier->notifier.notify = mptsas_cancel_notify;

                scsi_req_cancel_async(r, &notifier->notifier);

                goto reply_maybe_async;

            }

        }

        break;



    case MPI_SCSITASKMGMT_TASKTYPE_ABRT_TASK_SET:

    case MPI_SCSITASKMGMT_TASKTYPE_CLEAR_TASK_SET:

        status = mptsas_scsi_device_find(s, req->Bus, req->TargetID,

                                         req->LUN, &sdev);

        if (status) {

            reply.IOCStatus = status;

            goto out;

        }

        if (sdev->lun != req->LUN[1]) {

            reply.ResponseCode = MPI_SCSITASKMGMT_RSP_TM_INVALID_LUN;

            goto out;

        }



        reply_async = g_memdup(&reply, sizeof(MPIMsgSCSITaskMgmtReply));

        reply_async->IOCLogInfo = INT_MAX;



        count = 0;

        QTAILQ_FOREACH_SAFE(r, &sdev->requests, next, next) {

            if (r->hba_private) {

                MPTSASCancelNotifier *notifier;



                count++;

                notifier = g_new(MPTSASCancelNotifier, 1);

                notifier->s = s;

                notifier->reply = reply_async;

                notifier->notifier.notify = mptsas_cancel_notify;

                scsi_req_cancel_async(r, &notifier->notifier);

            }

        }



reply_maybe_async:

        if (reply_async->TerminationCount < count) {

            reply_async->IOCLogInfo = count;

            return;

        }


        reply.TerminationCount = count;

        break;



    case MPI_SCSITASKMGMT_TASKTYPE_LOGICAL_UNIT_RESET:

        status = mptsas_scsi_device_find(s, req->Bus, req->TargetID,

                                         req->LUN, &sdev);

        if (status) {

            reply.IOCStatus = status;

            goto out;

        }

        if (sdev->lun != req->LUN[1]) {

            reply.ResponseCode = MPI_SCSITASKMGMT_RSP_TM_INVALID_LUN;

            goto out;

        }

        qdev_reset_all(&sdev->qdev);

        break;



    case MPI_SCSITASKMGMT_TASKTYPE_TARGET_RESET:

        if (req->Bus != 0) {

            reply.IOCStatus = MPI_IOCSTATUS_SCSI_INVALID_BUS;

            goto out;

        }

        if (req->TargetID > s->max_devices) {

            reply.IOCStatus = MPI_IOCSTATUS_SCSI_INVALID_TARGETID;

            goto out;

        }



        QTAILQ_FOREACH(kid, &s->bus.qbus.children, sibling) {

            sdev = SCSI_DEVICE(kid->child);

            if (sdev->channel == 0 && sdev->id == req->TargetID) {

                qdev_reset_all(kid->child);

            }

        }

        break;



    case MPI_SCSITASKMGMT_TASKTYPE_RESET_BUS:

        qbus_reset_all(&s->bus.qbus);

        break;



    default:

        reply.ResponseCode = MPI_SCSITASKMGMT_RSP_TM_NOT_SUPPORTED;

        break;

    }



out:

    mptsas_fix_scsi_task_mgmt_reply_endianness(&reply);

    mptsas_post_reply(s, (MPIDefaultReply *)&reply);

}