static void clear_program(MpegTSContext *ts, unsigned int programid)

{

    int i;



    clear_avprogram(ts, programid);

    for(i=0; i<ts->nb_prg; i++)

        if(ts->prg[i].id == programid)

            ts->prg[i].nb_pids = 0;

}
