static void colored_fputs(int level, const char *str)

{

    if (!*str)

        return;



    if (use_color < 0) {

#if HAVE_SETCONSOLETEXTATTRIBUTE

        CONSOLE_SCREEN_BUFFER_INFO con_info;

        con = GetStdHandle(STD_ERROR_HANDLE);

        use_color = (con != INVALID_HANDLE_VALUE) && !getenv("NO_COLOR") &&

                    !getenv("AV_LOG_FORCE_NOCOLOR");

        if (use_color) {

            GetConsoleScreenBufferInfo(con, &con_info);

            attr_orig  = con_info.wAttributes;

            background = attr_orig & 0xF0;

        }

#elif HAVE_ISATTY

        use_color = !getenv("NO_COLOR") && !getenv("AV_LOG_FORCE_NOCOLOR") &&

                    (getenv("TERM") && isatty(2) ||

                     getenv("AV_LOG_FORCE_COLOR"));

        if (getenv("AV_LOG_FORCE_256COLOR"))

            use_color *= 256;

#else

        use_color = getenv("AV_LOG_FORCE_COLOR") && !getenv("NO_COLOR") &&

                   !getenv("AV_LOG_FORCE_NOCOLOR");

#endif

    }



#if HAVE_SETCONSOLETEXTATTRIBUTE

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
