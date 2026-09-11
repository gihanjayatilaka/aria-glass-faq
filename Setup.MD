# List of commands


1. Install Aria Companion https://facebookresearch.github.io/projectaria_tools/gen2/ark/companion-app/start
2. Login using your work meta account (https://forms.gle/2fckrA9DM4oAEsfZ9)
3. Connect the glasses to the phone app and record something
4. Connect the smart glasses to the PC using USB C
5. ```bash
   conda create -n aria python=3.11
   ```
6. ```bash
   conda activate aria
   ```
7. ```bash
   pip install projectaria-client-sdk
   ```
8. ```bash
   sudo apt install adb fastboot
   ```
9. ```bash
   aria_gen2 device list
   ```
10. ```bash
   aria_gen2 auth pair
   ```
11. Open the mobile phone app and authenticate
12. ```bash
    aria_gen2 recording list
    ```
13. ```bash
    aria_gen2 recording download -u 2c7b95d7-3534-43c7-bfe4-7db16a7ffa3d -o ./
    ```
14. ```bash
    aria_rerun_viewer --vrs videoName.vrs
    ```
