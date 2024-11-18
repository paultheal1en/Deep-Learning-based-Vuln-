int path_is_absolute(const char *path)

{

    const char *p;

#ifdef _WIN32

    /* specific case for names like: "\\.\d:" */

    if (*path == '/' || *path == '\\')

        return 1;

#endif

    p = strchr(path, ':');

    if (p)

        p++;

    else

        p = path;

#ifdef _WIN32

    return (*p == '/' || *p == '\\');

#else

    return (*p == '/');

#endif

}
