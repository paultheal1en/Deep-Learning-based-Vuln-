static void dump_regs(TCGContext *s)

{

    TCGTemp *ts;

    int i;

    char buf[64];



    for(i = 0; i < s->nb_temps; i++) {

        ts = &s->temps[i];

        printf("  %10s: ", tcg_get_arg_str_idx(s, buf, sizeof(buf), i));

        switch(ts->val_type) {

        case TEMP_VAL_REG:

            printf("%s", tcg_target_reg_names[ts->reg]);

            break;

        case TEMP_VAL_MEM:

            printf("%d(%s)", (int)ts->mem_offset, tcg_target_reg_names[ts->mem_reg]);

            break;

        case TEMP_VAL_CONST:

            printf("$0x%" TCG_PRIlx, ts->val);

            break;

        case TEMP_VAL_DEAD:

            printf("D");

            break;

        default:

            printf("???");

            break;

        }

        printf("\n");

    }



    for(i = 0; i < TCG_TARGET_NB_REGS; i++) {

        if (s->reg_to_temp[i] >= 0) {

            printf("%s: %s\n", 

                   tcg_target_reg_names[i], 

                   tcg_get_arg_str_idx(s, buf, sizeof(buf), s->reg_to_temp[i]));

        }

    }

}
