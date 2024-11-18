static int get_high_utility_cell(elbg_data *elbg)

{

    int i=0;

    /* Using linear search, do binary if it ever turns to be speed critical */

    int r = av_lfg_get(elbg->rand_state)%elbg->utility_inc[elbg->numCB-1] + 1;

    while (elbg->utility_inc[i] < r)

        i++;



    av_assert2(elbg->cells[i]);



    return i;

}
