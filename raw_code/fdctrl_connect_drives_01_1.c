static void fdctrl_connect_drives(FDCtrl *fdctrl)

{

    unsigned int i;

    FDrive *drive;



    for (i = 0; i < MAX_FD; i++) {

        drive = &fdctrl->drives[i];



        fd_init(drive);

        fd_revalidate(drive);

        if (drive->bs) {

            bdrv_set_removable(drive->bs, 1);

        }

    }

}
