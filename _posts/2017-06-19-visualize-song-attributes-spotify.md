---
layout: post
title:  "Visualizing song attributes from Spotify"
date:   2017-06-19
tags: spotify
image: /assets/article_images/2014-08-29-welcome-to-jekyll/desktop.JPG
comments: true
---

As a die-hard music fan and a long time Spotify user, I was excited to have discovered Spotify Web API a couple of days ago. I stumbled upon an analysis of Radiohead's gloomiest songs using a combination of audio features scraped from Spotify API and that gave me some great idea to do a statistical analysis of my favorite songs.

There's hardly a day gone by that I haven't used Spotify for at least half an hour, and that gave Spotify enough information to curate a playlist of my most streamed songs of 2016. The playlist contains 100 songs covering a wide spectrum of genres, and I was quite curious what insights I could learn about my musical preferences.
## Connecting to Spotify Web API
<br>
To get started with Spotify Web API, you have to first set up a developer account [here](https://developer.spotify.com/my-applications/#!/applications), register an application, and get your *Client ID* and *Client Secret*, both of which are needed to generate your access token and header value for your HTTP GET request.

{% highlight r %}
library(tidyverse)
library(httr)


clientID = 'xxxxxxxxxxxxxx'
clientSecret = 'xxxxxxxxxxxxxx'

# Get header value from Client ID and Client Secret
# Dependencies: tidyverse, httr
get_header_value <- function(client_id, client_secret) {
  token <- POST(
    'https://accounts.spotify.com/api/token',
    accept_json(),
    authenticate(clientID, secret),
    body = list(grant_type = 'client_credentials'),
    encode = 'form',
    verbose()
  ) %>% content %>% .$access_token
  header_value <- paste0('Bearer ', token)
  return(header_value)
}

header_value <- get_header_value(clientID, clientSecret)

{% endhighlight %}
