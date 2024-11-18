void tcg_profile_snapshot(TCGProfile *prof, bool counters, bool table)

{

    unsigned int i;



    for (i = 0; i < n_tcg_ctxs; i++) {

        const TCGProfile *orig = &tcg_ctxs[i]->prof;



        if (counters) {

            PROF_ADD(prof, orig, tb_count1);

            PROF_ADD(prof, orig, tb_count);

            PROF_ADD(prof, orig, op_count);

            PROF_MAX(prof, orig, op_count_max);

            PROF_ADD(prof, orig, temp_count);

            PROF_MAX(prof, orig, temp_count_max);

            PROF_ADD(prof, orig, del_op_count);

            PROF_ADD(prof, orig, code_in_len);

            PROF_ADD(prof, orig, code_out_len);

            PROF_ADD(prof, orig, search_out_len);

            PROF_ADD(prof, orig, interm_time);

            PROF_ADD(prof, orig, code_time);

            PROF_ADD(prof, orig, la_time);

            PROF_ADD(prof, orig, opt_time);

            PROF_ADD(prof, orig, restore_count);

            PROF_ADD(prof, orig, restore_time);

        }

        if (table) {

            int i;



            for (i = 0; i < NB_OPS; i++) {

                PROF_ADD(prof, orig, table_op_count[i]);

            }

        }

    }

}
