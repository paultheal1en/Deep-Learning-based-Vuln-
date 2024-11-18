static void add_pid_to_pmt(MpegTSContext *ts, unsigned int programid,

                           unsigned int pid)

{

    struct Program *p = get_program(ts, programid);

    int i;

    if (!p)

        return;



    if (p->nb_pids >= MAX_PIDS_PER_PROGRAM)

        return;



    for (i = 0; i < MAX_PIDS_PER_PROGRAM; i++)

        if (p->pids[i] == pid)

            return;



    p->pids[p->nb_pids++] = pid;

}
