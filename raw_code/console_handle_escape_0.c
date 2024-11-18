static void console_handle_escape(QemuConsole *s)

{

    int i;



    for (i=0; i<s->nb_esc_params; i++) {

        switch (s->esc_params[i]) {

            case 0: /* reset all console attributes to default */

                s->t_attrib = s->t_attrib_default;

                break;

            case 1:

                s->t_attrib.bold = 1;

                break;

            case 4:

                s->t_attrib.uline = 1;

                break;

            case 5:

                s->t_attrib.blink = 1;

                break;

            case 7:

                s->t_attrib.invers = 1;

                break;

            case 8:

                s->t_attrib.unvisible = 1;

                break;

            case 22:

                s->t_attrib.bold = 0;

                break;

            case 24:

                s->t_attrib.uline = 0;

                break;

            case 25:

                s->t_attrib.blink = 0;

                break;

            case 27:

                s->t_attrib.invers = 0;

                break;

            case 28:

                s->t_attrib.unvisible = 0;

                break;

            /* set foreground color */

            case 30:

                s->t_attrib.fgcol=COLOR_BLACK;

                break;

            case 31:

                s->t_attrib.fgcol=COLOR_RED;

                break;

            case 32:

                s->t_attrib.fgcol=COLOR_GREEN;

                break;

            case 33:

                s->t_attrib.fgcol=COLOR_YELLOW;

                break;

            case 34:

                s->t_attrib.fgcol=COLOR_BLUE;

                break;

            case 35:

                s->t_attrib.fgcol=COLOR_MAGENTA;

                break;

            case 36:

                s->t_attrib.fgcol=COLOR_CYAN;

                break;

            case 37:

                s->t_attrib.fgcol=COLOR_WHITE;

                break;

            /* set background color */

            case 40:

                s->t_attrib.bgcol=COLOR_BLACK;

                break;

            case 41:

                s->t_attrib.bgcol=COLOR_RED;

                break;

            case 42:

                s->t_attrib.bgcol=COLOR_GREEN;

                break;

            case 43:

                s->t_attrib.bgcol=COLOR_YELLOW;

                break;

            case 44:

                s->t_attrib.bgcol=COLOR_BLUE;

                break;

            case 45:

                s->t_attrib.bgcol=COLOR_MAGENTA;

                break;

            case 46:

                s->t_attrib.bgcol=COLOR_CYAN;

                break;

            case 47:

                s->t_attrib.bgcol=COLOR_WHITE;

                break;

        }

    }

}
