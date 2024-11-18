const char *path(const char *name)

{

    /* Only do absolute paths: quick and dirty, but should mostly be OK.

       Could do relative by tracking cwd. */

    if (!base || name[0] != '/')

	return name;



    return follow_path(base, name) ?: name;

}
