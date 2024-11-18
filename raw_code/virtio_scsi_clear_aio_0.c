static void virtio_scsi_clear_aio(VirtIOSCSI *s)

{

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    int i;



    if (s->ctrl_vring) {

        aio_set_event_notifier(s->ctx, &s->ctrl_vring->host_notifier,

                               false, NULL);

    }

    if (s->event_vring) {

        aio_set_event_notifier(s->ctx, &s->event_vring->host_notifier,

                               false, NULL);

    }

    if (s->cmd_vrings) {

        for (i = 0; i < vs->conf.num_queues && s->cmd_vrings[i]; i++) {

            aio_set_event_notifier(s->ctx, &s->cmd_vrings[i]->host_notifier,

                                   false, NULL);

        }

    }

}
