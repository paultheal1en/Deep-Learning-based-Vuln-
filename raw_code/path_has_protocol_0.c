static int path_has_protocol(const char *path)

{

#ifdef _WIN32

    if (is_windows_drive(path) ||

        is_windows_drive_prefix(path)) {

        return 0;

    }

#endif



    return strchr(path, ':') != NULL;

}
