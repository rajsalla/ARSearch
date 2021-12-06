/*
	Spatial by TEMPLATED
	templated.co @templatedco
	Released for free under the Creative Commons Attribution 3.0 license (templated.co/license)
*/

(function($) {
	console.log("hello");
	skel.breakpoints({
		xlarge:	'(max-width: 1680px)',
		large:	'(max-width: 1280px)',
		medium:	'(max-width: 980px)',
		small:	'(max-width: 736px)',
		xsmall:	'(max-width: 480px)'
	});

	$(function() {

		// var	$window = $(window),
		var $window = jQuery(window)
			$body = $('body');

		// Disable animations/transitions until the page has loaded.
			$body.addClass('is-loading');

			$('window').on('load', function() {
				window.setTimeout(function() {
					$body.removeClass('is-loading');
				}, 100);
			});

		// Fix: Placeholder polyfill.
			$('form').placeholder();

		// Prioritize "important" elements on medium.
			skel.on('+medium -medium', function() {
				$.prioritize(
					'.important\\28 medium\\29',
					skel.breakpoint('medium').active
				);
			});

		// Off-Canvas Navigation.

			// Navigation Panel Toggle.
				/*$('<a href="#navPanel" class="navPanelToggle"></a>')
					.appendTo($body);
				$(
					'<div id="navPanel">' +
						$('#nav').html() +
						'<a href="#navPanel" class="close"></a>' +
					'</div>'
				)
					.appendTo($body)
					.panel({
						delay: 500,
						hideOnClick: true,
						hideOnSwipe: true,
						resetScroll: true,
						resetForms: true,
						side: 'right'
					});*/

			// Fix: Remove transitions on WP<10 (poor/buggy performance).
				if (skel.vars.os == 'wp' && skel.vars.osVersion < 10)
					$('#navPanel')
						.css('transition', 'none');

			// submit query form
			$("#queryeditable").keypress(function(e){
				if(e.which == 13){
					$("#query").val($(this).text());
					$("form[name=search]").submit();
				}
				return true;
			});

			$(".container h2 svg").on("click", function(){
				$("#query").val($("#queryeditable").text());
				console.log($("#queryeditable").text());
				$("form[name=search]").submit();
			});
			$(".container h2 svg").hover(function(){
				console.log($("#queryeditable").text());
			});

			// reference form
			// $("form[name=reference]").on("submit", function(){
			// 	$.ajax({
			// 	  method: "POST",
			// 	  url: "/reference",
			// 	  data: { url: $("input[name=url]").val(), email: $("input[name=email]").val() }
			// 	})
			//   .done(function( msg ) {
			//     alert(msg );
			//   });
			// 	return false;
			// })
			console.log($("#queryeditable").text());
	});

})(jQuery);
