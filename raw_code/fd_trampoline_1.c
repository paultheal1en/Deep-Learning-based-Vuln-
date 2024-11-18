static gboolean fd_trampoline(GIOChannel *chan, GIOCondition cond, gpointer opaque)

{

    IOTrampoline *tramp = opaque;



    if (tramp->opaque == NULL) {

        return FALSE;

    }



    if ((cond & G_IO_IN) && tramp->fd_read) {

        tramp->fd_read(tramp->opaque);

    }



    if ((cond & G_IO_OUT) && tramp->fd_write) {

        tramp->fd_write(tramp->opaque);

    }



    return TRUE;

}
