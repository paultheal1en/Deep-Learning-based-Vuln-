void ff_mqc_init_contexts(MqcState *mqc)

{

    int i;

    memset(mqc->cx_states, 0, sizeof(mqc->cx_states));

    mqc->cx_states[MQC_CX_UNI] = 2 * 46;

    mqc->cx_states[MQC_CX_RL]  = 2 * 3;

    mqc->cx_states[0]          = 2 * 4;



    for (i = 0; i < 47; i++) {

        ff_mqc_qe[2 * i]     =

        ff_mqc_qe[2 * i + 1] = cx_states[i].qe;



        ff_mqc_nlps[2 * i]     = 2 * cx_states[i].nlps + cx_states[i].sw;

        ff_mqc_nlps[2 * i + 1] = 2 * cx_states[i].nlps + 1 - cx_states[i].sw;

        ff_mqc_nmps[2 * i]     = 2 * cx_states[i].nmps;

        ff_mqc_nmps[2 * i + 1] = 2 * cx_states[i].nmps + 1;

    }

}
