int ga_install_service(const char *path, const char *logfile)

{

    SC_HANDLE manager;

    SC_HANDLE service;

    TCHAR cmdline[MAX_PATH];



    if (GetModuleFileName(NULL, cmdline, MAX_PATH) == 0) {

        printf_win_error("No full path to service's executable");

        return EXIT_FAILURE;

    }



    _snprintf(cmdline, MAX_PATH - strlen(cmdline), "%s -d", cmdline);



    if (path) {

        _snprintf(cmdline, MAX_PATH - strlen(cmdline), "%s -p %s", cmdline, path);

    }

    if (logfile) {

        _snprintf(cmdline, MAX_PATH - strlen(cmdline), "%s -l %s -v",

            cmdline, logfile);

    }



    g_debug("service's cmdline: %s", cmdline);



    manager = OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);

    if (manager == NULL) {

        printf_win_error("No handle to service control manager");

        return EXIT_FAILURE;

    }



    service = CreateService(manager, QGA_SERVICE_NAME, QGA_SERVICE_DISPLAY_NAME,

        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START,

        SERVICE_ERROR_NORMAL, cmdline, NULL, NULL, NULL, NULL, NULL);



    if (service) {

        SERVICE_DESCRIPTION desc = { (char *)QGA_SERVICE_DESCRIPTION };

        ChangeServiceConfig2(service, SERVICE_CONFIG_DESCRIPTION, &desc);



        printf("Service was installed successfully.\n");

    } else {

        printf_win_error("Failed to install service");

    }



    CloseServiceHandle(service);

    CloseServiceHandle(manager);



    return (service == NULL);

}
