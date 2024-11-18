static GuestPCIAddress *get_pci_info(char *guid, Error **errp)

{

    HDEVINFO dev_info;

    SP_DEVINFO_DATA dev_info_data;

    DWORD size = 0;

    int i;

    char dev_name[MAX_PATH];

    char *buffer = NULL;

    GuestPCIAddress *pci = NULL;

    char *name = g_strdup(&guid[4]);



    if (!QueryDosDevice(name, dev_name, ARRAY_SIZE(dev_name))) {

        error_setg_win32(errp, GetLastError(), "failed to get dos device name");

        goto out;

    }



    dev_info = SetupDiGetClassDevs(&GUID_DEVINTERFACE_VOLUME, 0, 0,

                                   DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);

    if (dev_info == INVALID_HANDLE_VALUE) {

        error_setg_win32(errp, GetLastError(), "failed to get devices tree");

        goto out;

    }



    dev_info_data.cbSize = sizeof(SP_DEVINFO_DATA);

    for (i = 0; SetupDiEnumDeviceInfo(dev_info, i, &dev_info_data); i++) {

        DWORD addr, bus, slot, func, dev, data, size2;

        while (!SetupDiGetDeviceRegistryProperty(dev_info, &dev_info_data,

                                            SPDRP_PHYSICAL_DEVICE_OBJECT_NAME,

                                            &data, (PBYTE)buffer, size,

                                            &size2)) {

            size = MAX(size, size2);

            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {

                g_free(buffer);

                /* Double the size to avoid problems on

                 * W2k MBCS systems per KB 888609.

                 * https://support.microsoft.com/en-us/kb/259695 */

                buffer = g_malloc(size * 2);

            } else {

                error_setg_win32(errp, GetLastError(),

                        "failed to get device name");

                goto out;

            }

        }



        if (g_strcmp0(buffer, dev_name)) {

            continue;

        }



        /* There is no need to allocate buffer in the next functions. The size

         * is known and ULONG according to

         * https://support.microsoft.com/en-us/kb/253232

         * https://msdn.microsoft.com/en-us/library/windows/hardware/ff543095(v=vs.85).aspx

         */

        if (!SetupDiGetDeviceRegistryProperty(dev_info, &dev_info_data,

                   SPDRP_BUSNUMBER, &data, (PBYTE)&bus, size, NULL)) {

            break;

        }



        /* The function retrieves the device's address. This value will be

         * transformed into device function and number */

        if (!SetupDiGetDeviceRegistryProperty(dev_info, &dev_info_data,

                   SPDRP_ADDRESS, &data, (PBYTE)&addr, size, NULL)) {

            break;

        }



        /* This call returns UINumber of DEVICE_CAPABILITIES structure.

         * This number is typically a user-perceived slot number. */

        if (!SetupDiGetDeviceRegistryProperty(dev_info, &dev_info_data,

                   SPDRP_UI_NUMBER, &data, (PBYTE)&slot, size, NULL)) {

            break;

        }



        /* SetupApi gives us the same information as driver with

         * IoGetDeviceProperty. According to Microsoft

         * https://support.microsoft.com/en-us/kb/253232

         * FunctionNumber = (USHORT)((propertyAddress) & 0x0000FFFF);

         * DeviceNumber = (USHORT)(((propertyAddress) >> 16) & 0x0000FFFF);

         * SPDRP_ADDRESS is propertyAddress, so we do the same.*/



        func = addr & 0x0000FFFF;

        dev = (addr >> 16) & 0x0000FFFF;

        pci = g_malloc0(sizeof(*pci));

        pci->domain = dev;

        pci->slot = slot;

        pci->function = func;

        pci->bus = bus;

        break;

    }

out:

    g_free(buffer);

    g_free(name);

    return pci;

}
