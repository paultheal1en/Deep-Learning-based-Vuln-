static void add_pid_to_pmt(MpegTSContext *ts, unsigned int programid, unsigned int pid)

{

    int i;

    struct Program *p = NULL;

    for(i=0; i<ts->nb_prg; i++) {

        if(ts->prg[i].id == programid) {

            p = &ts->prg[i];

            break;

        }

    }

    if(!p)

        return;



    if(p->nb_pids >= MAX_PIDS_PER_PROGRAM)

        return;

    p->pids[p->nb_pids++] = pid;

}
