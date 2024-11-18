static void qxl_log_cmd_draw(PCIQXLDevice *qxl, QXLDrawable *draw, int group_id)

{

    fprintf(stderr, ": surface_id %d type %s effect %s",

            draw->surface_id,

            qxl_name(qxl_draw_type, draw->type),

            qxl_name(qxl_draw_effect, draw->effect));

    switch (draw->type) {

    case QXL_DRAW_COPY:

        qxl_log_cmd_draw_copy(qxl, &draw->u.copy, group_id);

        break;

    }

}
