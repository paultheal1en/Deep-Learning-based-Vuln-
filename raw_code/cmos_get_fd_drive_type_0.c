static int cmos_get_fd_drive_type(FloppyDriveType fd0)

{

    int val;



    switch (fd0) {

    case FLOPPY_DRIVE_TYPE_144:

        /* 1.44 Mb 3"5 drive */

        val = 4;

        break;

    case FLOPPY_DRIVE_TYPE_288:

        /* 2.88 Mb 3"5 drive */

        val = 5;

        break;

    case FLOPPY_DRIVE_TYPE_120:

        /* 1.2 Mb 5"5 drive */

        val = 2;

        break;

    case FLOPPY_DRIVE_TYPE_NONE:

    default:

        val = 0;

        break;

    }

    return val;

}
