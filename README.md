Audience repo


#####visdom usage

Start the server (probably in a screen or tmux) from the command line:

> visdom

Visdom now can be accessed by going to http://localhost:8097 in your browser, or your own host address if specified.

The visdom command is equivalent to running python -m visdom.server.

If the above does not work, try using an SSH tunnel to your server by adding the following line to your local ~/.ssh/config: LocalForward 127.0.0.1:8097 127.0.0.1:8097.# image2image_stego
