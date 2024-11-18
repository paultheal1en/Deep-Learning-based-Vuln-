static void proxy_rewinddir(FsContext *ctx, V9fsFidOpenState *fs)

{

    rewinddir(fs->dir);

}
