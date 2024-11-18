static inline void ide_abort_command(IDEState *s)

{

    ide_transfer_stop(s);

    s->status = READY_STAT | ERR_STAT;

    s->error = ABRT_ERR;

}
