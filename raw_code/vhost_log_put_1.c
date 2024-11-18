static void vhost_log_put(struct vhost_dev *dev, bool sync)

{

    struct vhost_log *log = dev->log;



    if (!log) {

        return;

    }

    dev->log = NULL;

    dev->log_size = 0;



    --log->refcnt;

    if (log->refcnt == 0) {

        /* Sync only the range covered by the old log */

        if (dev->log_size && sync) {

            vhost_log_sync_range(dev, 0, dev->log_size * VHOST_LOG_CHUNK - 1);

        }



        if (vhost_log == log) {

            g_free(log->log);

            vhost_log = NULL;

        } else if (vhost_log_shm == log) {

            qemu_memfd_free(log->log, log->size * sizeof(*(log->log)),

                            log->fd);

            vhost_log_shm = NULL;

        }



        g_free(log);

    }

}
