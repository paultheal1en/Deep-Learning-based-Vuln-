int qemu_ftruncate64(int fd, int64_t length)

{

    LARGE_INTEGER li;

    LONG high;

    HANDLE h;

    BOOL res;



    if ((GetVersion() & 0x80000000UL) && (length >> 32) != 0)

	return -1;



    h = (HANDLE)_get_osfhandle(fd);



    /* get current position, ftruncate do not change position */

    li.HighPart = 0;

    li.LowPart = SetFilePointer (h, 0, &li.HighPart, FILE_CURRENT);

    if (li.LowPart == 0xffffffffUL && GetLastError() != NO_ERROR)

	return -1;



    high = length >> 32;

    if (!SetFilePointer(h, (DWORD) length, &high, FILE_BEGIN))

	return -1;

    res = SetEndOfFile(h);



    /* back to old position */

    SetFilePointer(h, li.LowPart, &li.HighPart, FILE_BEGIN);

    return res ? 0 : -1;

}
