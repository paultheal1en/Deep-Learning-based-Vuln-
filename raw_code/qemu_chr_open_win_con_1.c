static int qemu_chr_open_win_con(QemuOpts *opts, CharDriverState **chr)

{

    return qemu_chr_open_win_file(GetStdHandle(STD_OUTPUT_HANDLE), chr);

}
