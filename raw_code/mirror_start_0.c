void mirror_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *target, const char *replaces,

                  int64_t speed, uint32_t granularity, int64_t buf_size,

                  MirrorSyncMode mode, BlockMirrorBackingMode backing_mode,

                  BlockdevOnError on_source_error,

                  BlockdevOnError on_target_error,

                  bool unmap,

                  BlockCompletionFunc *cb,

                  void *opaque, Error **errp)

{

    bool is_none_mode;

    BlockDriverState *base;



    if (mode == MIRROR_SYNC_MODE_INCREMENTAL) {

        error_setg(errp, "Sync mode 'incremental' not supported");

        return;

    }

    is_none_mode = mode == MIRROR_SYNC_MODE_NONE;

    base = mode == MIRROR_SYNC_MODE_TOP ? backing_bs(bs) : NULL;

    mirror_start_job(job_id, bs, target, replaces,

                     speed, granularity, buf_size, backing_mode,

                     on_source_error, on_target_error, unmap, cb, opaque, errp,

                     &mirror_job_driver, is_none_mode, base);

}
