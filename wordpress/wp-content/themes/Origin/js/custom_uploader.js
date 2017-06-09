(function($){
	$(document).ready(function() {
		$('.upload_image_button').click( function( event ) {
			var $this_el = $(this);

			event.preventDefault();

			et_pb_file_frame = wp.media.frames.et_pb_file_frame = wp.media({
				title: $this_el.data( 'choose' ),
				library: {
					type: 'image'
				},
				button: {
					text: $this_el.data( 'update' ),
				},
				multiple: false
			});

			et_pb_file_frame.on( 'select', function() {
				var attachment = et_pb_file_frame.state().get('selection').first().toJSON();

				$this_el.prev( 'input' ).val( attachment.url );
			});

			et_pb_file_frame.open();
		} );
	} );
} )(jQuery);