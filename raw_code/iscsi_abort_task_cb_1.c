iscsi_abort_task_cb(struct iscsi_context *iscsi, int status, void *command_data,

                    void *private_data)

{

    IscsiAIOCB *acb = (IscsiAIOCB *)private_data;



    scsi_free_scsi_task(acb->task);

    acb->task = NULL;

}
