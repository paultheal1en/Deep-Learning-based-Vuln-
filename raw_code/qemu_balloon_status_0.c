static int qemu_balloon_status(BalloonInfo *info)

{

    if (!balloon_stat_fn) {

        return 0;

    }

    balloon_stat_fn(balloon_opaque, info);

    return 1;

}
