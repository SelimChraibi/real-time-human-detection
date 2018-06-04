pip install git+git://github.com/scikit-learn/scikit-learn.git

pip install google_images_download

# # install chromedriver (used for batch download image from google image)

# pip install chromedriver_installer \
#     --install-option="--chromedriver-version=2.10" \
#     --install-option="--chromedriver-checksums=4fecc99b066cb1a346035bf022607104,058cd8b7b4b9688507701b5e648fd821"

# cp $(which chromedriver) ./utilities/


# install sbt
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt


# install imagemagick
sudo apt-get install imagemagick

# get open_images_downloader repo
git clone https://github.com/dnuffer/open_images_downloader.git

# compile, create the target, unzip it and add it to PATH
# open_images_downloader --download-300k --resize-mode FillCrop --resize-compression-quality 50 --nolog-to-stdout &