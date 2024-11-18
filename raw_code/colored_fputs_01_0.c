static void colored_fputs(int level, int tint, const char *str)

{

    if (!*str)

        return;



    if (use_color < 0)

        check_color_terminal();



#if defined(_WIN32) && !defined(__MINGW32CE__) && HAVE_SETCONSOLETEXTATTRIBUTE

    if (use_color && level != AV_LOG_INFO/8)

        SetConsoleTextAttribute(con, background | color[level]);

    fputs(str, stderr);

    if (use_color && level != AV_LOG_INFO/8)

        SetConsoleTextAttribute(con, attr_orig);

#else

    if (use_color == 1 && level != AV_LOG_INFO/8) {

        fprintf(stderr,

                "\033[%d;3%dm%s\033[0m",

                (color[level] >> 4) & 15,

                color[level] & 15,

                str);

    } else if (tint && use_color == 256) {

        fprintf(stderr,

                "\033[48;5;%dm\033[38;5;%dm%s\033[0m",

                (color[level] >> 16) & 0xff,

                tint,

                str);

    } else if (use_color == 256 && level != AV_LOG_INFO/8) {

        fprintf(stderr,

                "\033[48;5;%dm\033[38;5;%dm%s\033[0m",

                (color[level] >> 16) & 0xff,

                (color[level] >> 8) & 0xff,

                str);

    } else

        fputs(str, stderr);

#endif



}
