static void lsi_command_complete(SCSIBus *bus, int reason, uint32_t tag,

                                 uint32_t arg)

{

    LSIState *s = DO_UPCAST(LSIState, dev.qdev, bus->qbus.parent);

    int out;



    out = (s->sstat1 & PHASE_MASK) == PHASE_DO;

    if (reason == SCSI_REASON_DONE) {

        DPRINTF("Command complete status=%d\n", (int)arg);

        s->status = arg;

        s->command_complete = 2;

        if (s->waiting && s->dbc != 0) {

            /* Raise phase mismatch for short transfers.  */

            lsi_bad_phase(s, out, PHASE_ST);

        } else {

            lsi_set_phase(s, PHASE_ST);

        }



        qemu_free(s->current);

        s->current = NULL;



        lsi_resume_script(s);

        return;

    }



    if (s->waiting == 1 || !s->current || tag != s->current->tag ||

        (lsi_irq_on_rsl(s) && !(s->scntl1 & LSI_SCNTL1_CON))) {

        if (lsi_queue_tag(s, tag, arg))

            return;

    }



    /* host adapter (re)connected */

    DPRINTF("Data ready tag=0x%x len=%d\n", tag, arg);

    s->current->dma_len = arg;

    s->command_complete = 1;

    if (!s->waiting)

        return;

    if (s->waiting == 1 || s->dbc == 0) {

        lsi_resume_script(s);

    } else {

        lsi_do_dma(s, out);

    }

}
