static int verify_expr(AVExpr *e)

{

    if (!e) return 0;

    switch (e->type) {

        case e_value:

        case e_const: return 1;

        case e_func0:

        case e_func1:

        case e_squish:

        case e_ld:

        case e_gauss:

        case e_isnan:

        case e_floor:

        case e_ceil:

        case e_trunc:

        case e_sqrt:

        case e_not:

        case e_random:

            return verify_expr(e->param[0]) && !e->param[2];

        case e_taylor:

            return verify_expr(e->param[0]) && verify_expr(e->param[1])

                   && (!e->param[2] || verify_expr(e->param[2]));

        default: return verify_expr(e->param[0]) && verify_expr(e->param[1]) && !e->param[2];

    }

}
