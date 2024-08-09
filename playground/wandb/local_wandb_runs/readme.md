# Runnin wanbd local server
ref: https://chatgpt.com/c/f8c4b5fe-bec5-4f53-a503-00b91a69c8b4
ref: https://docs.wandb.ai/guides/hosting/self-managed/basic-setup#3-generate-a-license <--- this is better

## Run Wandb Server (so local website works)
Start wanbd local (e.g., laptop sever):
```bash
# - Start Wandb Server locally (M1 Mac)
# Run a Docker container for the W&B server with a persistent volume, port mapping, and automatic restart.
docker run -d --restart always --platform linux/amd64 -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
# - Start Wandb Server (other archs)
# docker run -d --restart always -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local

# - Create your local login (username "user" and password "local") -- not used for local browser
# rm ~/.netrc
echo -e "machine localhost\nlogin user\npassword locallocal" > ~/.netrc
cat ~/.netrc
# output
machine localhost
login user
password locallocal
```
Now go to browser and open http://localhost:8080/ and you should see a wandb service running. 
Even though you put the `~/.netrc` credentials, you need to still sign up (confusingly). 
So go to the local host sign up and put the same credentials as above for from your `~/.netrc`, to potential issues though this is unknown to me). 
You can save it in your 1password if you want. 

Now display your ready to display your runs if! Run one if you don't have a run data

conda activate 
pip install wandb

wandb login --host=http://localhost:8080
export WANDB_API_KEY=from local host login page, click copy
get license: https://deploy.wandb.ai/65ff2ab8-c7b4-4075-92ef-af41e1f740a1/licenses


Tips recall docker:
```bash
# check container
docker ps
docker stop <container_id>
docker rm <container_di>
```

# Appendix

## Inspecting the volume of the container 
Inspect wandb container:
```bash
# If docker container is
docker run -d --restart always --platform linux/amd64 -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local
# To inspect volume named wandb, do (why the name is wandb and not wandb/local):
docker volume inspect wandb
# [
#     {
#         "CreatedAt": "2024-08-09T01:15:30Z",
#         "Driver": "local",
#         "Labels": null,
#         "Mountpoint": "/var/lib/docker/volumes/wandb/_data",
#         "Name": "wandb",
#         "Options": null,
#         "Scope": "local"
#     }
# Note: above is not local filesystem
# To remove volume do:
docker stop wandb-local
docker rm wandb-local
docker ps
docker volume ls
docker volume rm wandb
docker volume ls
]```