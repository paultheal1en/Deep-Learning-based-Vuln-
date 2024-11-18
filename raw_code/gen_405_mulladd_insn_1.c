static always_inline void gen_405_mulladd_insn (DisasContext *ctx,

                                                int opc2, int opc3,

                                                int ra, int rb, int rt, int Rc)

{

    gen_op_load_gpr_T0(ra);

    gen_op_load_gpr_T1(rb);

    switch (opc3 & 0x0D) {

    case 0x05:

        /* macchw    - macchw.    - macchwo   - macchwo.   */

        /* macchws   - macchws.   - macchwso  - macchwso.  */

        /* nmacchw   - nmacchw.   - nmacchwo  - nmacchwo.  */

        /* nmacchws  - nmacchws.  - nmacchwso - nmacchwso. */

        /* mulchw - mulchw. */

        gen_op_405_mulchw();

        break;

    case 0x04:

        /* macchwu   - macchwu.   - macchwuo  - macchwuo.  */

        /* macchwsu  - macchwsu.  - macchwsuo - macchwsuo. */

        /* mulchwu - mulchwu. */

        gen_op_405_mulchwu();

        break;

    case 0x01:

        /* machhw    - machhw.    - machhwo   - machhwo.   */

        /* machhws   - machhws.   - machhwso  - machhwso.  */

        /* nmachhw   - nmachhw.   - nmachhwo  - nmachhwo.  */

        /* nmachhws  - nmachhws.  - nmachhwso - nmachhwso. */

        /* mulhhw - mulhhw. */

        gen_op_405_mulhhw();

        break;

    case 0x00:

        /* machhwu   - machhwu.   - machhwuo  - machhwuo.  */

        /* machhwsu  - machhwsu.  - machhwsuo - machhwsuo. */

        /* mulhhwu - mulhhwu. */

        gen_op_405_mulhhwu();

        break;

    case 0x0D:

        /* maclhw    - maclhw.    - maclhwo   - maclhwo.   */

        /* maclhws   - maclhws.   - maclhwso  - maclhwso.  */

        /* nmaclhw   - nmaclhw.   - nmaclhwo  - nmaclhwo.  */

        /* nmaclhws  - nmaclhws.  - nmaclhwso - nmaclhwso. */

        /* mullhw - mullhw. */

        gen_op_405_mullhw();

        break;

    case 0x0C:

        /* maclhwu   - maclhwu.   - maclhwuo  - maclhwuo.  */

        /* maclhwsu  - maclhwsu.  - maclhwsuo - maclhwsuo. */

        /* mullhwu - mullhwu. */

        gen_op_405_mullhwu();

        break;

    }

    if (opc2 & 0x02) {

        /* nmultiply-and-accumulate (0x0E) */

        gen_op_neg();

    }

    if (opc2 & 0x04) {

        /* (n)multiply-and-accumulate (0x0C - 0x0E) */

        gen_op_load_gpr_T2(rt);

        gen_op_move_T1_T0();

        gen_op_405_add_T0_T2();

    }

    if (opc3 & 0x10) {

        /* Check overflow */

        if (opc3 & 0x01)

            gen_op_405_check_ov();

        else

            gen_op_405_check_ovu();

    }

    if (opc3 & 0x02) {

        /* Saturate */

        if (opc3 & 0x01)

            gen_op_405_check_sat();

        else

            gen_op_405_check_satu();

    }

    gen_op_store_T0_gpr(rt);

    if (unlikely(Rc) != 0) {

        /* Update Rc0 */

        gen_set_Rc0(ctx);

    }

}
