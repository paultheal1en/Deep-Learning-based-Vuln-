void qga_vss_fsfreeze(int *nr_volume, Error **errp, bool freeze)

{

    const char *func_name = freeze ? "requester_freeze" : "requester_thaw";

    QGAVSSRequesterFunc func;

    ErrorSet errset = {

        .error_setg_win32 = error_setg_win32,

        .errp = errp,

    };



    g_assert(errp);             /* requester.cpp requires it */

    func = (QGAVSSRequesterFunc)GetProcAddress(provider_lib, func_name);

    if (!func) {

        error_setg_win32(errp, GetLastError(), "failed to load %s from %s",

                         func_name, QGA_VSS_DLL);

        return;

    }



    func(nr_volume, &errset);

}
