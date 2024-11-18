static void mux_chr_accept_input(CharDriverState *chr)

{

    int m = chr->focus;

    MuxDriver *d = chr->opaque;



    while (d->prod != d->cons &&

           d->chr_can_read[m] &&

           d->chr_can_read[m](d->ext_opaque[m])) {

        d->chr_read[m](d->ext_opaque[m],

                       &d->buffer[d->cons++ & MUX_BUFFER_MASK], 1);

    }

}
